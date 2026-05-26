// bvh_writer.cpp - see bvh_writer.h for the high-level strategy.
//
// Math sketch
// ───────────
//   joint_params = PT @ model_params[:pt_cols]            // [n_joints*7]
//   q_euler[j]   = qz(rz) * qy(ry) * qx(rx)               // MHR convention: R=Rz·Ry·Rx
//   q_local[j]   = pre[j] * q_euler[j]                    // MHR per-joint local rotation
//   g_q[j]       = g_q[parent] * q_local[j]               // MHR FK
//
// For each BVH joint b with matched MHR index m and BVH parent bp:
//   q_target = g_q_mhr[m]                                  // MHR world orientation of m
//   q_local_bvh[b] = inv(g_q_bvh[bp]) * q_target
//   g_q_bvh[b] = q_target
// Decompose q_local_bvh[b] to ZXY (body) or ZYX (root) Euler angles for the
// channel order in body.bvh ("Zrotation Xrotation Yrotation" or "Zrotation
// Yrotation Xrotation").  Going via global quats avoids the rest-pose alignment
// (R_align) issue: the BVH chain starts identity at the root joint and inherits
// MHR's global orientation joint-by-joint.

#include "bvh_writer.h"
#include "fast_sam_3dbody.h"
#include "mhr_joint_table.h"

extern "C" {
#include "ModelLoader/model_loader_transform_joints.h"
}

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr float RAD2DEG  = 57.29577951308232f;
constexpr float POS_SCALE = 100.0f;             // metres → centimetres

// ─── quaternion helpers (XYZW) ──────────────────────────────────────────────

// Hamilton product: r = a * b  (so when rotating v, b is applied first, then a).
inline void qmul(float* r, const float* a, const float* b)
{
    r[0] = b[0]*a[3] + b[3]*a[0] + b[2]*a[1] - b[1]*a[2];
    r[1] = b[1]*a[3] - b[2]*a[0] + b[3]*a[1] + b[0]*a[2];
    r[2] = b[2]*a[3] + b[1]*a[0] - b[0]*a[1] + b[3]*a[2];
    r[3] = b[3]*a[3] - b[0]*a[0] - b[1]*a[1] - b[2]*a[2];
}

inline void qconj(float* r, const float* a)
{
    r[0] = -a[0]; r[1] = -a[1]; r[2] = -a[2]; r[3] = a[3];
}

inline void qrot(float* out, const float* q, const float* v)
{
    float qx=q[0], qy=q[1], qz=q[2], qw=q[3];
    float vx=v[0], vy=v[1], vz=v[2];
    float tx = 2.f*(qy*vz - qz*vy);
    float ty = 2.f*(qz*vx - qx*vz);
    float tz = 2.f*(qx*vy - qy*vx);
    out[0] = vx + qw*tx + (qy*tz - qz*ty);
    out[1] = vy + qw*ty + (qz*tx - qx*tz);
    out[2] = vz + qw*tz + (qx*ty - qy*tx);
}

// MHR Euler (XYZ intrinsic, R = Rz·Ry·Rx) → quaternion.
// Identical to mhr_euler_xyz_to_quat in model_loader_transform_joints.c.
inline void euler_mhr_to_quat(float ex, float ey, float ez, float* q)
{
    float hx=ex*0.5f, hy=ey*0.5f, hz=ez*0.5f;
    float qx[4] = { sinf(hx), 0.f,      0.f,      cosf(hx) };
    float qy[4] = { 0.f,      sinf(hy), 0.f,      cosf(hy) };
    float qz[4] = { 0.f,      0.f,      sinf(hz), cosf(hz) };
    float t[4];
    qmul(t, qz, qy);
    qmul(q, t,  qx);
}

// Quaternion (XYZW) → 3×3 row-major rotation matrix.
inline void quat_to_mat3(const float* q, float m[9])
{
    float x=q[0], y=q[1], z=q[2], w=q[3];
    float xx=x*x, yy=y*y, zz=z*z;
    float xy=x*y, xz=x*z, yz=y*z;
    float wx=w*x, wy=w*y, wz=w*z;
    m[0] = 1 - 2*(yy + zz); m[1] = 2*(xy - wz);     m[2] = 2*(xz + wy);
    m[3] = 2*(xy + wz);     m[4] = 1 - 2*(xx + zz); m[5] = 2*(yz - wx);
    m[6] = 2*(xz - wy);     m[7] = 2*(yz + wx);     m[8] = 1 - 2*(xx + yy);
}

// Decompose R = Rz(a) · Rx(b) · Ry(c)  (BVH non-root channels: Zrot Xrot Yrot).
//   M[2][1] = sin(b)
//   M[2][0] = -cos(b) sin(c)   →  c = atan2(-M[2][0], M[2][2])
//   M[0][1] = -cos(b) sin(a)   →  a = atan2(-M[0][1], M[1][1])
inline void mat3_to_zxy(const float m[9], float& a, float& b, float& c)
{
    float s = std::max(-1.f, std::min(1.f, m[7]));    // m[2][1]
    b = asinf(s);
    if (fabsf(m[7]) < 0.99999f) {
        c = atan2f(-m[6], m[8]);                       // -m[2][0], m[2][2]
        a = atan2f(-m[1], m[4]);                       // -m[0][1],  m[1][1]
    } else {
        // Gimbal: lock Y to zero, solve Z from the remaining DOF.
        c = 0.f;
        a = atan2f(m[3], m[0]);
    }
}

// Decompose R = Rz(a) · Ry(b) · Rx(c)  (BVH root channels: Zrot Yrot Xrot).
//   M[2][0] = -sin(b)
//   M[2][1] = cos(b) sin(c)
//   M[2][2] = cos(b) cos(c)
//   M[1][0] = cos(b) sin(a)
//   M[0][0] = cos(b) cos(a)
inline void mat3_to_zyx(const float m[9], float& a, float& b, float& c)
{
    float s = std::max(-1.f, std::min(1.f, -m[6]));   // -m[2][0]
    b = asinf(s);
    if (fabsf(m[6]) < 0.99999f) {
        c = atan2f(m[7], m[8]);                        //  m[2][1], m[2][2]
        a = atan2f(m[3], m[0]);                        //  m[1][0], m[0][0]
    } else {
        c = 0.f;
        a = atan2f(-m[1], m[4]);
    }
}

// ─── BVH template parsing ───────────────────────────────────────────────────

// Per-joint info collected from the BVH template, in declaration order.
struct ParsedJoint {
    std::string name;            // empty for End Site
    int         parent_idx;      // -1 for root
    float       offset[3];
    int         channel_offset;  // -1 if no channels
    int         n_channels;
};

// Tiny ASCII parser. Tokens are whitespace-separated; we keep a small ad-hoc
// state machine for the curly-brace hierarchy.
bool parse_bvh_template(const std::string& path,
                        std::string& hierarchy_text,
                        std::vector<ParsedJoint>& out_joints,
                        int& out_total_channels)
{
    FILE* f = fopen(path.c_str(), "r");
    if (!f) {
        fprintf(stderr, "[BVHWriter] cannot open template '%s'\n", path.c_str());
        return false;
    }

    out_joints.clear();
    hierarchy_text.clear();
    out_total_channels = 0;

    std::vector<int> stack;          // parent indices
    int  channel_cursor = 0;
    bool found_motion  = false;

    enum PendingKind { PEND_NONE, PEND_NAMED, PEND_END_SITE };
    PendingKind pending_kind = PEND_NONE;
    std::string pending_name;

    char line[4096];
    while (fgets(line, sizeof(line), f))
    {
        const char* p = line;
        while (*p == ' ' || *p == '\t') ++p;

        if (strncmp(p, "MOTION", 6) == 0 &&
            (p[6]=='\n' || p[6]=='\r' || p[6]=='\0' || p[6]==' ' || p[6]=='\t'))
        {
            found_motion = true;
            break;
        }

        hierarchy_text += line;

        // ROOT name / JOINT name
        if ((strncmp(p,"ROOT", 4)==0 && (p[4]==' '||p[4]=='\t')) ||
            (strncmp(p,"JOINT",5)==0 && (p[5]==' '||p[5]=='\t')))
        {
            const char* q = p + ((p[0]=='R') ? 4 : 5);
            while (*q==' '||*q=='\t') ++q;
            const char* qend = q;
            while (*qend && *qend!=' ' && *qend!='\t' && *qend!='\r' && *qend!='\n') ++qend;
            pending_kind = PEND_NAMED;
            pending_name.assign(q, qend - q);
            continue;
        }

        if (strncmp(p, "End Site", 8) == 0) {
            pending_kind = PEND_END_SITE;
            pending_name.clear();
            continue;
        }

        if (*p == '{') {
            // create the pending joint now so OFFSET/CHANNELS attach to it
            ParsedJoint j;
            j.name           = (pending_kind == PEND_NAMED) ? pending_name : std::string();
            j.parent_idx     = stack.empty() ? -1 : stack.back();
            j.offset[0]=j.offset[1]=j.offset[2]=0.f;
            j.channel_offset = -1;
            j.n_channels     = 0;
            int idx = (int)out_joints.size();
            out_joints.push_back(j);
            stack.push_back(idx);
            pending_kind = PEND_NONE;
            pending_name.clear();
            continue;
        }

        if (*p == '}') {
            if (!stack.empty()) stack.pop_back();
            continue;
        }

        if (strncmp(p, "OFFSET", 6) == 0 && (p[6]==' '||p[6]=='\t')) {
            if (!stack.empty()) {
                ParsedJoint& j = out_joints[stack.back()];
                sscanf(p + 6, "%f %f %f", &j.offset[0], &j.offset[1], &j.offset[2]);
            }
            continue;
        }

        if (strncmp(p, "CHANNELS", 8) == 0 && (p[8]==' '||p[8]=='\t')) {
            int n_ch = 0;
            sscanf(p + 8, "%d", &n_ch);
            if (!stack.empty() && n_ch > 0) {
                ParsedJoint& j = out_joints[stack.back()];
                j.channel_offset = channel_cursor;
                j.n_channels     = n_ch;
                channel_cursor  += n_ch;
            } else if (n_ch > 0) {
                channel_cursor += n_ch;     // safety net; shouldn't happen
            }
            continue;
        }
    }

    fclose(f);

    if (!found_motion) {
        fprintf(stderr, "[BVHWriter] template '%s' has no MOTION block\n", path.c_str());
        return false;
    }

    out_total_channels = channel_cursor;
    return true;
}

}  // namespace

// ─── BVH rest-pose world positions ──────────────────────────────────────────

void BVHWriter::compute_bvh_rest_positions()
{
    // BVH rest pose: all rotations zero, root translation zero.
    // World position of each joint = parent's world pos + own OFFSET.
    for (auto& j : joints_) {
        if (j.parent < 0) {
            j.rest_world[0] = j.offset[0];
            j.rest_world[1] = j.offset[1];
            j.rest_world[2] = j.offset[2];
        } else {
            const BvhJoint& p = joints_[j.parent];
            j.rest_world[0] = p.rest_world[0] + j.offset[0];
            j.rest_world[1] = p.rest_world[1] + j.offset[1];
            j.rest_world[2] = p.rest_world[2] + j.offset[2];
        }
    }
}

// ─── Explicit BVH-name → MHR-name table ────────────────────────────────────
//
// Built by hand against body.bvh (MakeHuman) and the MHR joint names exported
// from mhr_model.pt (see scripts/build_joint_table.py).  Names not listed here
// are left unmapped — their BVH channels stay at zero, which is what we want
// for joints MHR doesn't model (BVH face details, buttocks, BVH toes, etc.).
namespace {
struct NameMap { const char* bvh; const char* mhr; };

constexpr NameMap NAME_MAP[] = {
    // ── Spine ────────────────────────────────────────────────────────────
    { "hip",        "root"      },
    { "abdomen",    "c_spine1"  },
    { "chest",      "c_spine3"  },
    { "neck",       "c_neck"    },
    { "head",       "c_head"    },
    { "jaw",        "c_jaw"     },

    // ── Left arm ─────────────────────────────────────────────────────────
    { "lCollar",    "l_clavicle"},
    { "lShldr",     "l_uparm"   },
    { "lForeArm",   "l_lowarm"  },
    { "lHand",      "l_wrist"   },

    // ── Right arm ────────────────────────────────────────────────────────
    { "rCollar",    "r_clavicle"},
    { "rShldr",     "r_uparm"   },
    { "rForeArm",   "r_lowarm"  },
    { "rHand",      "r_wrist"   },

    // ── Left leg (BVH lButtock skipped — no analogue in MHR) ─────────────
    { "lThigh",     "l_upleg"   },
    { "lShin",      "l_lowleg"  },
    { "lFoot",      "l_foot"    },

    // ── Right leg ────────────────────────────────────────────────────────
    { "rThigh",     "r_upleg"   },
    { "rShin",      "r_lowleg"  },
    { "rFoot",      "r_foot"    },

    // ── Left hand fingers (BVH finger2..5 = index/middle/ring/pinky;
    //    finger1 = thumb).  Metacarpals are BVH-only and stay at zero. ────
    { "finger1-2.l", "l_thumb2" },
    { "finger1-3.l", "l_thumb3" },
    { "rthumb",      "l_thumb1" },  // intentionally lower-case 'rthumb' as found in body.bvh

    { "finger2-1.l", "l_index1" },
    { "finger2-2.l", "l_index2" },
    { "finger2-3.l", "l_index3" },

    { "finger3-1.l", "l_middle1"},
    { "finger3-2.l", "l_middle2"},
    { "finger3-3.l", "l_middle3"},

    { "finger4-1.l", "l_ring1"  },
    { "finger4-2.l", "l_ring2"  },
    { "finger4-3.l", "l_ring3"  },

    { "finger5-1.l", "l_pinky1" },
    { "finger5-2.l", "l_pinky2" },
    { "finger5-3.l", "l_pinky3" },

    // ── Right hand fingers ───────────────────────────────────────────────
    { "finger1-2.r", "r_thumb2" },
    { "finger1-3.r", "r_thumb3" },

    { "finger2-1.r", "r_index1" },
    { "finger2-2.r", "r_index2" },
    { "finger2-3.r", "r_index3" },

    { "finger3-1.r", "r_middle1"},
    { "finger3-2.r", "r_middle2"},
    { "finger3-3.r", "r_middle3"},

    { "finger4-1.r", "r_ring1"  },
    { "finger4-2.r", "r_ring2"  },
    { "finger4-3.r", "r_ring3"  },

    { "finger5-1.r", "r_pinky1" },
    { "finger5-2.r", "r_pinky2" },
    { "finger5-3.r", "r_pinky3" },
};
}  // namespace

bool BVHWriter::match_bvh_to_mhr()
{
    if (!lbs_) return false;
    const int  nj = lbs_->n_joints;
    const int* parents = lbs_->joint_parents;
    const float* offs  = lbs_->joint_offsets;
    const float* pre   = lbs_->joint_prerotations;

    // FK over MHR with zero pose. Stores rest joint quaternions in
    // q_global_mhr_rest_ for use in build_frame_row().
    q_global_mhr_rest_.assign((size_t)nj * 4, 0.f);
    std::vector<float> g_t(nj * 3, 0.f);
    for (int j = 0; j < nj; ++j) {
        const float* p_q = pre + j*4;          // q_local = pre (zero euler ⇒ q_euler = identity)
        int parent = parents[j];
        if (parent < 0) {
            g_t[j*3+0] = offs[j*3+0];
            g_t[j*3+1] = offs[j*3+1];
            g_t[j*3+2] = offs[j*3+2];
            memcpy(&q_global_mhr_rest_[j*4], p_q, 4*sizeof(float));
        } else {
            float rt[3];
            qrot(rt, &q_global_mhr_rest_[parent*4], offs + j*3);
            g_t[j*3+0] = g_t[parent*3+0] + rt[0];
            g_t[j*3+1] = g_t[parent*3+1] + rt[1];
            g_t[j*3+2] = g_t[parent*3+2] + rt[2];
            qmul(&q_global_mhr_rest_[j*4], &q_global_mhr_rest_[parent*4], p_q);
        }
    }

    // The MHR internal frame already uses +Y up (the LBS-output flip is just
    // for the camera-facing display frame).  Keep g_t as-is; we'll determine
    // whether any axis flip is needed by inspecting the rest-pose layout below.
    std::vector<float> mhr_world_bvh_frame(nj * 3);
    for (int j = 0; j < nj; ++j) {
        mhr_world_bvh_frame[j*3+0] = g_t[j*3+0];
        mhr_world_bvh_frame[j*3+1] = g_t[j*3+1];
        mhr_world_bvh_frame[j*3+2] = g_t[j*3+2];
    }

    // The MHR skeleton has body_world(j0) as a virtual root at the body's TOP
    // and root(j1) as the hip — that's the one body.bvh's "hip" corresponds to.
    // Translate so MHR's "root" lands on BVH's hip OFFSET.
    int mhr_hip = -1;
    for (int j = 0; j < nj; ++j) {
        if (strcmp(mhr_joint_table::NAMES[j], "root") == 0) { mhr_hip = j; break; }
    }
    if (mhr_hip < 0) {
        fprintf(stderr, "[BVHWriter] could not find 'root' in MHR joint table\n");
        return false;
    }
    float dx = joints_[root_idx_].rest_world[0] - mhr_world_bvh_frame[mhr_hip*3+0];
    float dy = joints_[root_idx_].rest_world[1] - mhr_world_bvh_frame[mhr_hip*3+1];
    float dz = joints_[root_idx_].rest_world[2] - mhr_world_bvh_frame[mhr_hip*3+2];
    for (int j = 0; j < nj; ++j) {
        mhr_world_bvh_frame[j*3+0] += dx;
        mhr_world_bvh_frame[j*3+1] += dy;
        mhr_world_bvh_frame[j*3+2] += dz;
    }

    // Optional rest-pose dump (set BVH_WRITER_DUMP=1 to enable).
    if (const char* env = getenv("BVH_WRITER_DUMP")) {
        if (env[0] == '1') {
            FILE* dbg = fopen("/tmp/bvh_writer_skeletons.csv", "w");
            if (dbg) {
                fprintf(dbg, "side,name,parent,x,y,z\n");
                for (size_t i = 0; i < joints_.size(); ++i) {
                    const BvhJoint& j = joints_[i];
                    if (j.name.empty()) continue;
                    fprintf(dbg, "bvh,%s,%d,%.6f,%.6f,%.6f\n",
                            j.name.c_str(), j.parent,
                            j.rest_world[0], j.rest_world[1], j.rest_world[2]);
                }
                for (int j = 0; j < nj; ++j) {
                    fprintf(dbg, "mhr,j%d,%d,%.6f,%.6f,%.6f\n",
                            j, parents[j],
                            mhr_world_bvh_frame[j*3+0],
                            mhr_world_bvh_frame[j*3+1],
                            mhr_world_bvh_frame[j*3+2]);
                }
                fclose(dbg);
                fprintf(stderr, "[BVHWriter] wrote /tmp/bvh_writer_skeletons.csv\n");
            }
        }
    }

    // Build BVH-name → MHR-index lookup using the explicit NAME_MAP table.
    // For each mapping, print the rest-pose Euclidean distance between the
    // two skeletons so we can spot mis-pairings.
    std::unordered_map<std::string, int> mhr_name_to_idx;
    for (int j = 0; j < mhr_joint_table::N_JOINTS; ++j)
        mhr_name_to_idx[mhr_joint_table::NAMES[j]] = j;

    std::unordered_map<std::string, const char*> bvh_to_mhr_name;
    for (const auto& nm : NAME_MAP) bvh_to_mhr_name[nm.bvh] = nm.mhr;

    int n_named = 0;
    int n_matched = 0;
    float worst_d = 0.f; std::string worst_name;

    for (size_t i = 0; i < joints_.size(); ++i) {
        BvhJoint& bj = joints_[i];
        bj.mhr_idx = -1;
        if (bj.name.empty() || bj.n_channels == 0) continue;
        ++n_named;

        auto it = bvh_to_mhr_name.find(bj.name);
        if (it == bvh_to_mhr_name.end()) continue;

        auto mi = mhr_name_to_idx.find(it->second);
        if (mi == mhr_name_to_idx.end()) {
            fprintf(stderr, "[BVHWriter] WARNING: mapping target '%s' not in MHR table\n",
                    it->second);
            continue;
        }

        int mhr_idx = mi->second;
        bj.mhr_idx = mhr_idx;
        ++n_matched;

        float dxj = mhr_world_bvh_frame[mhr_idx*3+0] - bj.rest_world[0];
        float dyj = mhr_world_bvh_frame[mhr_idx*3+1] - bj.rest_world[1];
        float dzj = mhr_world_bvh_frame[mhr_idx*3+2] - bj.rest_world[2];
        float d   = sqrtf(dxj*dxj + dyj*dyj + dzj*dzj);
        if (d > worst_d) { worst_d = d; worst_name = bj.name; }
    }

    fprintf(stderr,
            "[BVHWriter] explicit-name match: %d / %d named BVH joints "
            "(worst rest-pose offset: %s d=%.2f cm)\n",
            n_matched, n_named, worst_name.c_str(), worst_d);

    return n_matched > 0;
}

// ─── Per-frame MHR-side FK ─────────────────────────────────────────────────

void BVHWriter::compute_per_frame_mhr_state(const fsb::MHRResult& r)
{
    const int nj   = lbs_->n_joints;
    const int npc  = lbs_->pt_cols;
    const float* PT = lbs_->PT;
    const float* pre = lbs_->joint_prerotations;
    const int*  parents = lbs_->joint_parents;

    // joint_params = PT · model_params (PT input width = pt_cols; model_params is 204).
    const int take = std::min(npc, 204);
    for (int row = 0; row < nj * 7; ++row) {
        const float* prow = PT + (size_t)row * npc;
        float acc = 0.f;
        for (int k = 0; k < take; ++k) acc += prow[k] * r.mhr_model_params[k];
        joint_params_[row] = acc;
    }

    // q_local[j] = pre[j] * euler(rx, ry, rz)
    // g_q[j]     = g_q[parent] * q_local[j]
    for (int j = 0; j < nj; ++j) {
        const float* jp = &joint_params_[j * 7];
        float q_euler[4];
        euler_mhr_to_quat(jp[3], jp[4], jp[5], q_euler);
        qmul(&q_local_[j*4], pre + j*4, q_euler);
        int p = parents[j];
        if (p < 0) memcpy(&q_global_mhr_[j*4], &q_local_[j*4], 4*sizeof(float));
        else       qmul(&q_global_mhr_[j*4], &q_global_mhr_[p*4], &q_local_[j*4]);
    }
}

// ─── Build one BVH motion row ───────────────────────────────────────────────
//
// Math
// ────
//   delta_mhr[j]  = g_q_mhr[j] · conj(g_q_mhr_rest[j])    (world delta in MHR frame)
//   delta_bvh[j]  = S · delta_mhr[j] · S⁻¹                (world delta in BVH frame)
//   R_local_bvh[b] = inv(delta_bvh[bp]) · delta_bvh[b]
//                  = S · inv(delta_mhr[bp]) · delta_mhr[b] · S⁻¹
//
//   In quaternion form, conjugation by S = diag(1,-1,-1) is just (qx,-qy,-qz,qw).
//
// Why deltas?
//   BVH rest is identity at every joint; MHR rest has prerotations baked in.
//   We must compare CURRENT rotation to REST in MHR before assigning to BVH.

void BVHWriter::build_frame_row(const fsb::MHRResult& r, float* row)
{
    memset(row, 0, total_channels_ * sizeof(float));
    if (!lbs_) return;

    compute_per_frame_mhr_state(r);

    auto delta_mhr = [&](int m, float* out)
    {
        // out = q_global_mhr_[m] * conj(q_global_mhr_rest_[m])
        float inv_rest[4]; qconj(inv_rest, &q_global_mhr_rest_[m * 4]);
        qmul(out, &q_global_mhr_[m * 4], inv_rest);
    };

    for (size_t bi = 0; bi < joints_.size(); ++bi) {
        BvhJoint& bj = joints_[bi];
        if (bj.n_channels == 0) continue;
        if (bj.mhr_idx   <  0)  continue;

        float q_delta_self[4]; delta_mhr(bj.mhr_idx, q_delta_self);

        float q_local_bvh[4];
        if (bj.parent < 0) {
            memcpy(q_local_bvh, q_delta_self, 4*sizeof(float));
        } else {
            int pj = bj.parent, mp = -1;
            while (pj >= 0) {
                if (joints_[pj].mhr_idx >= 0 && joints_[pj].n_channels > 0) {
                    mp = joints_[pj].mhr_idx; break;
                }
                pj = joints_[pj].parent;
            }
            if (mp < 0) {
                memcpy(q_local_bvh, q_delta_self, 4*sizeof(float));
            } else {
                float q_delta_par[4]; delta_mhr(mp, q_delta_par);
                float inv_par[4]; qconj(inv_par, q_delta_par);
                qmul(q_local_bvh, inv_par, q_delta_self);
            }
        }

        // MHR internal frame and BVH frame agree on all three axes (verified
        // by rest-pose dump); no quaternion sign flip needed.

        float m[9]; quat_to_mat3(q_local_bvh, m);

        if (bj.is_root && bj.n_channels == 6) {
            // Root channels: Xpos Ypos Zpos  Zrot Yrot Xrot.
            row[bj.channel_offset + 0] = r.pred_cam_t[0] * POS_SCALE;
            row[bj.channel_offset + 1] = r.pred_cam_t[1] * POS_SCALE;
            row[bj.channel_offset + 2] = r.pred_cam_t[2] * POS_SCALE;
            float a, b, c;
            mat3_to_zyx(m, a, b, c);
            row[bj.channel_offset + 3] = a * RAD2DEG;
            row[bj.channel_offset + 4] = b * RAD2DEG;
            row[bj.channel_offset + 5] = c * RAD2DEG;
        } else if (bj.n_channels == 3) {
            // Body-joint channels: Zrot Xrot Yrot.
            float a, b, c;
            mat3_to_zxy(m, a, b, c);
            row[bj.channel_offset + 0] = a * RAD2DEG;
            row[bj.channel_offset + 1] = b * RAD2DEG;
            row[bj.channel_offset + 2] = c * RAD2DEG;
        }
    }
}

// ─── BVHWriter::open ────────────────────────────────────────────────────────

bool BVHWriter::open(const std::string& template_path,
                     const std::string& out_path,
                     float              frame_time,
                     const std::string& lbs_path)
{
    // Parse the BVH template.
    std::string hierarchy;
    std::vector<ParsedJoint> parsed;
    int total_ch = 0;
    if (!parse_bvh_template(template_path, hierarchy, parsed, total_ch))
        return false;
    if (total_ch <= 0) {
        fprintf(stderr, "[BVHWriter] template parsed 0 channels\n");
        return false;
    }

    // Transfer parsed joints into our internal struct.
    joints_.clear();
    joints_.reserve(parsed.size());
    root_idx_ = -1;
    for (size_t i = 0; i < parsed.size(); ++i) {
        const ParsedJoint& pj = parsed[i];
        BvhJoint bj;
        bj.name           = pj.name;
        bj.parent         = pj.parent_idx;
        bj.offset[0]      = pj.offset[0];
        bj.offset[1]      = pj.offset[1];
        bj.offset[2]      = pj.offset[2];
        bj.channel_offset = pj.channel_offset;
        bj.n_channels     = pj.n_channels;
        bj.rest_world[0]=bj.rest_world[1]=bj.rest_world[2]=0.f;
        bj.mhr_idx        = -1;
        bj.is_root        = (pj.parent_idx < 0);
        joints_.push_back(bj);
        if (bj.is_root && root_idx_ < 0) root_idx_ = (int)i;
    }
    if (root_idx_ < 0) {
        fprintf(stderr, "[BVHWriter] template has no root joint\n");
        return false;
    }

    compute_bvh_rest_positions();

    // Load the MHR LBS data (PT, joint_offsets, joint_prerotations, joint_parents).
    if (lbs_path.empty()) {
        fprintf(stderr, "[BVHWriter] lbs_path is empty — cannot match BVH joints\n");
        return false;
    }
    lbs_ = mhr_lbs_load(lbs_path.c_str());
    if (!lbs_) {
        fprintf(stderr, "[BVHWriter] mhr_lbs_load('%s') failed\n", lbs_path.c_str());
        return false;
    }
    // Match BVH joint names to MHR joint indices.
    if (!match_bvh_to_mhr()) {
        fprintf(stderr, "[BVHWriter] failed to match any BVH joint to MHR\n");
        mhr_lbs_free(lbs_); lbs_ = nullptr;
        return false;
    }

    // Allocate scratch buffers.
    joint_params_.assign((size_t)lbs_->n_joints * 7, 0.f);
    q_local_.assign     ((size_t)lbs_->n_joints * 4, 0.f);
    q_global_mhr_.assign((size_t)lbs_->n_joints * 4, 0.f);

    // Open output and write the hierarchy.
    file_ = fopen(out_path.c_str(), "wb");
    if (!file_) {
        fprintf(stderr, "[BVHWriter] cannot open output '%s'\n", out_path.c_str());
        mhr_lbs_free(lbs_); lbs_ = nullptr;
        return false;
    }

    total_channels_ = total_ch;
    frame_time_     = frame_time;
    frame_count_    = 0;

    fwrite(hierarchy.data(), 1, hierarchy.size(), file_);
    fprintf(file_, "MOTION\n");
    frames_pos_ = ftell(file_);
    fprintf(file_, "Frames: %10d\n", 0);
    fprintf(file_, "Frame Time: %.6f\n", frame_time_);
    return true;
}

// ─── BVHWriter::write_frame ────────────────────────────────────────────────

void BVHWriter::write_frame(const std::vector<fsb::MHRResult>& results)
{
    if (!file_ || !lbs_) return;

    // One contiguous row, large enough for the template (body.bvh is ~498 chans).
    std::vector<float> row(total_channels_, 0.f);
    if (!results.empty())
        build_frame_row(results[0], row.data());

    for (int i = 0; i < total_channels_; ++i) {
        if (i > 0) fputc(' ', file_);
        if (row[i] == 0.f)        fputc('0', file_);
        else                      fprintf(file_, "%.4f", row[i]);
    }
    fputc('\n', file_);
    ++frame_count_;
}

// ─── BVHWriter::close ──────────────────────────────────────────────────────

void BVHWriter::close()
{
    if (!file_) return;
    fseek(file_, frames_pos_, SEEK_SET);
    fprintf(file_, "Frames: %10d\n", frame_count_);
    fclose(file_);
    file_ = nullptr;
    printf("[BVHWriter] wrote %d frames\n", frame_count_);

    if (lbs_) { mhr_lbs_free(lbs_); lbs_ = nullptr; }
}
