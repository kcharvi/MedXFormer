README - MedXFormer: Cross-Specialty Disease Diagnosis for Improved Clinical Outcomes
====================================================================================

Top Level Folders for Guidance
------------------------------
/Report
    └── charviku_bhanucha_final_report.pdf

/Video
    ├── video.mp4
    └── MedXFormer_Presentation.pptx

/Code
    ├── run.sh
    ├── inference.py
    ├── requirements.txt
    ├── MedXFormer_Dataset_Creation.ipynb
    ├── MedXFormer_Multiple_LoRA-v3.ipynb
    ├── MedXFormer_Multiple_LoRA_on_Filtered_Dataset.ipynb
    └── MedXFormer_Multiple_LoRA_Inferences.ipynb


Summary of Sub-Directory /Code Contents
-----------------------------
**Folders**
- **medxformer_images_for_inferences**: Contains images for inference testing, used by `run.sh`.
- **medxformer_v3**: Stores weights of trained adapters for various datasets and PEFT techniques.

**Key Files**
- **MedXFormer_Dataset_Creation.ipynb**: Notebook for dataset analysis and preparation.
- **MedXFormer_Multiple_LoRA-v3.ipynb**: Handles training and evaluation of adapters on unfiltered data.
- **MedXFormer_Multiple_LoRA_on_Filtered_Dataset.ipynb**: Similar to the above, but operates on filtered data.
- **MedXFormer_Multiple_LoRA_Inferences.ipynb**: Performs inference with trained adapters, displays the image, actual label, predicted label, and confidence score for the inference results.
- **run.sh**: A Bash script enabling quick inference. This script sets up a virtual environment, installs necessary dependencies, and allows selecting models for testing. It fetches images from `medxformer_images_for_inferences` and uses adapter weights from `medxformer_v3`.

**How to Run run.sh**
1. Ensure you have `bash` and `python` installed.
2. Navigate to the directory containing `run.sh` (./Code).
3. Run `bash run.sh` using command: ./run.sh
4. Follow on-screen prompts to select the model and proceed with inference.

**inference.py**: A helper script invoked by `run.sh`.
**requirements.txt**: Lists required dependencies.

**Adapter Model Weights (medxformer_v3)**
- Organized by dataset (e.g., Brain Tumor, Skin Cancer) and PEFT method (e.g., LoRA, LoHA).
- Each subdirectory includes:
  - Model weights in `.safetensors` format.
  - Configuration files.
  - Training arguments.
  - Metadata.
