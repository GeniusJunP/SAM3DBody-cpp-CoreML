#pragma once
// bvh_writer.h - BVH export for the MHR body-pose pipeline.
//
// Strategy
// ────────
//   • Load the MHR LBS data (body_model.lbs) so we have PT, joint_offsets,
//     joint_prerotations, joint_parents.
//   • Parse the BVH hierarchy template (default ./body.bvh).
//   • At open(), match each BVH joint name to an MHR joint index by comparing
//     rest-pose world positions (BVH OFFSET chain vs. MHR FK with zero pose).
//   • At write_frame(), compute joint_params = PT · model_params[6:6+pt_cols] for
//     the current person, form each joint's quaternion q_local = pre · euler,
//     decompose to the BVH-channel Euler triple (ZXY for body joints, ZYX for the
//     root) and emit a frame row.
//
// Hand fingers, jaw, and any other joints that resolve to a valid MHR index are
// animated automatically — there is no manually-curated joint list.
//
// Usage
// ─────
//   BVHWriter w;
//   if (!w.open("body.bvh", "out.bvh", 1.f/30.f, "./onnx/body_model.lbs")) ...
//   for (const auto& frame : ...) w.write_frame(frame);
//   w.close();

#include <string>
#include <vector>

struct MHR_LBS_Data;             // GraphicsEngine/ModelLoader/model_loader_transform_joints.h
namespace fsb { struct MHRResult; }

class BVHWriter
{
public:
    bool open(const std::string& template_path,
              const std::string& out_path,
              float              frame_time = 1.0f / 30.0f,
              const std::string& lbs_path   = "");

    void write_frame(const std::vector<fsb::MHRResult>& results);
    void close();

    bool is_open()           const { return file_ != nullptr; }
    int  frame_count()       const { return frame_count_;     }
    int  channels_per_frame()const { return total_channels_; }

    BVHWriter()  = default;
    ~BVHWriter() { if (is_open()) close(); }
    BVHWriter(const BVHWriter&)            = delete;
    BVHWriter& operator=(const BVHWriter&) = delete;

private:
    // ── BVH-side joint metadata (built once at open()) ──────────────────────
    struct BvhJoint {
        std::string name;            // empty for End Site nodes
        int         parent;          // -1 for root
        float       offset[3];       // OFFSET from parent (cm), in parent frame
        int         channel_offset;  // first channel index in the frame row, -1 if no channels
        int         n_channels;      // 0, 3, or 6
        // World-space rest pose (BVH file rest = all rotations zero, root pos = 0)
        float       rest_world[3];
        // Resolved MHR joint index (-1 if no match)
        int         mhr_idx;
        bool        is_root;
    };

    // ── Persistent state ────────────────────────────────────────────────────
    FILE*  file_                 = nullptr;
    long   frames_pos_           = 0;
    int    frame_count_          = 0;
    int    total_channels_       = 0;
    float  frame_time_           = 1.0f / 30.0f;

    MHR_LBS_Data*         lbs_   = nullptr;
    std::vector<BvhJoint> joints_;       // includes End Site entries with n_channels=0
    int                   root_idx_ = -1;

    // ── Reusable per-frame scratch buffers ─────────────────────────────────
    std::vector<float> joint_params_;       // [n_joints * 7]   PT · model_params
    std::vector<float> q_local_;            // [n_joints * 4]   pre · q_euler
    std::vector<float> q_global_mhr_;       // [n_joints * 4]   per-frame FK
    std::vector<float> q_global_mhr_rest_;  // [n_joints * 4]   FK with zero pose

    // Helpers
    bool parse_template(const std::string& path, std::string& hierarchy_text);
    void compute_bvh_rest_positions();
    bool match_bvh_to_mhr();
    void compute_per_frame_mhr_state(const fsb::MHRResult& r);
    void build_frame_row(const fsb::MHRResult& r, float* row);
};
