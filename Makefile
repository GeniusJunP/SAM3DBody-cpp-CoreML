BACKBONE_SIZE ?= 512

.PHONY: all models build clean

all: models build

models:
	@echo "========================================"
	@echo " Exporting Backbone (Size: $(BACKBONE_SIZE))"
	@echo "========================================"
	cd coreml_export && .venv311/bin/python export_coreml_backbone.py --size $(BACKBONE_SIZE)
	cd coreml_export && .venv311/bin/python export_coreml_decoder.py --size $(BACKBONE_SIZE)
	@echo "Compiling CoreML packages to mlmodelc..."
	xcrun coremlcompiler compile ./coreml_export/backbone_coreml.mlpackage ./coreml_export/
	xcrun coremlcompiler compile ./coreml_export/decoder_coreml.mlpackage ./coreml_export/
	@echo "Models ready."

build:
	@echo "========================================"
	@echo " Building C++ Engine (Size: $(BACKBONE_SIZE))"
	@echo "========================================"
	cmake -S . -B build -DFSB_BACKBONE_SIZE=$(BACKBONE_SIZE) -DFSB_COREML=ON
	cmake --build build -j

clean:
	rm -rf build
	rm -rf coreml_export/backbone_coreml.mlmodelc
