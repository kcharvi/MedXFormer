import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from peft import PeftModel, PeftConfig
from PIL import Image
import os
import json

def load_mappings(MAPPINGS_PATH):
    with open(MAPPINGS_PATH, 'r') as f:
        mappings = json.load(f)
    label2id = mappings['label2id']
    id2label = mappings['id2label']
    
    return label2id, id2label

def load_model(MODEL_PATH, ADAPTER_PATH, num_labels):
    model = ViTForImageClassification.from_pretrained(MODEL_PATH, num_labels=num_labels)
    peft_config = PeftConfig.from_pretrained(ADAPTER_PATH)
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    processor = ViTImageProcessor.from_pretrained(MODEL_PATH)    
    return model, processor

def classify_image(model, processor, image_path, id2label):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)
    top_prob, top_predid = torch.topk(probabilities, k=2)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return id2label[str(prediction)], top_prob, top_predid

def print_classification_results(true_label, prediction, top_prob, top_predid, id2label):
    print(f"True Label: {true_label}")
    print(f"Predicted Label: {prediction}")
    print("Confidence Scores:")
    for i in range(top_prob.shape[1]):
        print(f"  {str(id2label[str(top_predid[0][i].item())])}: {top_prob[0][i].item():.4f}")
    print("-" * 40)

def brain_tumor_classification(MODEL_PATH, ADAPTER_PATH, IMAGES_FOLDER, MAPPINGS_PATH):
    label2id, id2label = load_mappings(MAPPINGS_PATH)
    num_labels = len(id2label)
    model, processor = load_model(MODEL_PATH, ADAPTER_PATH, num_labels)
    model.eval()
    for tumor_type in os.listdir(IMAGES_FOLDER):
        tumor_path = os.path.join(IMAGES_FOLDER, tumor_type)
        if not os.path.isdir(tumor_path):
            continue
        print(f"\nClassifying images in {tumor_type} folder:")
        for image_name in os.listdir(tumor_path):
            image_path = os.path.join(tumor_path, image_name)
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                continue
            true_label = tumor_type
            prediction, top_prob, top_predid = classify_image(model, processor, image_path, id2label)
            print_classification_results(true_label, prediction, top_prob, top_predid, id2label)

def diabetic_retinopathy_classification(MODEL_PATH, ADAPTER_PATH, IMAGES_FOLDER, MAPPINGS_PATH):
    label2id, id2label = load_mappings(MAPPINGS_PATH)
    num_labels = len(id2label)
    model, processor = load_model(MODEL_PATH, ADAPTER_PATH, num_labels)
    model.eval()
    for diabetic_type in os.listdir(IMAGES_FOLDER):
        diabetic_path = os.path.join(IMAGES_FOLDER, diabetic_type)
        if not os.path.isdir(diabetic_path):
            continue
        print(f"\nClassifying images in {diabetic_type} folder:")
        for image_name in os.listdir(diabetic_path):
            image_path = os.path.join(diabetic_path, image_name)
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                continue
            true_label = diabetic_type
            prediction, top_prob, top_predid = classify_image(model, processor, image_path, id2label)
            print_classification_results(true_label, prediction, top_prob, top_predid, id2label)

def kidney_disease_classification(MODEL_PATH, ADAPTER_PATH, IMAGES_FOLDER, MAPPINGS_PATH):
    label2id, id2label = load_mappings(MAPPINGS_PATH)
    num_labels = len(id2label)
    model, processor = load_model(MODEL_PATH, ADAPTER_PATH, num_labels)
    model.eval()
    for kidney_type in os.listdir(IMAGES_FOLDER):
        kidney_path = os.path.join(IMAGES_FOLDER, kidney_type)
        if not os.path.isdir(kidney_path):
            continue
        print(f"\nClassifying images in {kidney_type} folder:")
        for image_name in os.listdir(kidney_path):
            image_path = os.path.join(kidney_path, image_name)
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                continue
            true_label = kidney_type
            prediction, top_prob, top_predid = classify_image(model, processor, image_path, id2label)
            print_classification_results(true_label, prediction, top_prob, top_predid, id2label)

def retina_oct_classification(MODEL_PATH, ADAPTER_PATH, IMAGES_FOLDER, MAPPINGS_PATH):
    label2id, id2label = load_mappings(MAPPINGS_PATH)
    num_labels = len(id2label)
    model, processor = load_model(MODEL_PATH, ADAPTER_PATH, num_labels)
    model.eval()
    for retina_type in os.listdir(IMAGES_FOLDER):
        retina_path = os.path.join(IMAGES_FOLDER, retina_type)
        if not os.path.isdir(retina_path):
            continue
        print(f"\nClassifying images in {retina_type} folder:")
        for image_name in os.listdir(retina_path):
            image_path = os.path.join(retina_path, image_name)
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                continue
            true_label = retina_type
            prediction, top_prob, top_predid = classify_image(model, processor, image_path, id2label)
            print_classification_results(true_label, prediction, top_prob, top_predid, id2label)

def skin_cancer_classification(MODEL_PATH, ADAPTER_PATH, IMAGES_FOLDER, MAPPINGS_PATH):
    label2id, id2label = load_mappings(MAPPINGS_PATH)
    num_labels = len(id2label)
    model, processor = load_model(MODEL_PATH, ADAPTER_PATH, num_labels)
    model.eval()
    for skin_type in os.listdir(IMAGES_FOLDER):
        skin_path = os.path.join(IMAGES_FOLDER, skin_type)
        if not os.path.isdir(skin_path):
            continue
        print(f"\nClassifying images in {skin_type} folder:")
        for image_name in os.listdir(skin_path):
            image_path = os.path.join(skin_path, image_name)
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                continue
            true_label = skin_type
            prediction, top_prob, top_predid = classify_image(model, processor, image_path, id2label)
            print_classification_results(true_label, prediction, top_prob, top_predid, id2label)

def main():
    while True:
        print("Welcome to the Image Classification Tool!")
        print("Please choose one of the following options:")
        print("\n1. Brain Tumor Classification")
        print("2. Diabetic Retinopathy Classification")
        print("3. Kidney Disease Classification")
        print("4. Retina OCT Classification")
        print("5. Skin Cancer Classification")
        print("6. Exit")

        choice = input("\nEnter your choice (1-6): ")
        MODEL_PATH = "google/vit-base-patch16-224-in21k" 

        if choice == '1':
            ADAPTER_PATH = "./medxformer_v3/BRAIN_TUMOR_ADAPTER_MODEL_loha"
            IMAGES_FOLDER = "./medxformer_images_for_inferences/brain"
            MAPPINGS_PATH = "./medxformer_v3/BRAIN_TUMOR_ADAPTER_MODEL_loha/mappings.json"
            print(f"Starting Brain Tumor Classification for {IMAGES_FOLDER}")
            brain_tumor_classification(MODEL_PATH, ADAPTER_PATH, IMAGES_FOLDER, MAPPINGS_PATH)
        
        elif choice == '2':
            ADAPTER_PATH = "./medxformer_v3/DIABETIC_ADAPTER_MODEL_loha"
            IMAGES_FOLDER = "./medxformer_images_for_inferences/diabetic"
            MAPPINGS_PATH = "./medxformer_v3/DIABETIC_ADAPTER_MODEL_loha/mappings.json"
            print(f"Starting Diabetic Retinopathy Classification for {IMAGES_FOLDER}")
            diabetic_retinopathy_classification(MODEL_PATH, ADAPTER_PATH, IMAGES_FOLDER, MAPPINGS_PATH)
        
        elif choice == '3':
            ADAPTER_PATH = "./medxformer_v3/KIDNEY_ADAPTER_MODEL_lora"
            IMAGES_FOLDER = "./medxformer_images_for_inferences/kidney"
            MAPPINGS_PATH = "./medxformer_v3/KIDNEY_ADAPTER_MODEL_lora/mappings.json"
            print(f"Starting Kidney Disease Classification for {IMAGES_FOLDER}")
            kidney_disease_classification(MODEL_PATH, ADAPTER_PATH, IMAGES_FOLDER, MAPPINGS_PATH)
        
        elif choice == '4':
            ADAPTER_PATH = "./medxformer_v3/RETINA_ADAPTER_MODEL_loha"
            IMAGES_FOLDER = "./medxformer_images_for_inferences/retina"
            MAPPINGS_PATH = "./medxformer_v3/RETINA_ADAPTER_MODEL_loha/mappings.json"
            print(f"Starting Retina OCT Classification for {IMAGES_FOLDER}")
            retina_oct_classification(MODEL_PATH, ADAPTER_PATH, IMAGES_FOLDER, MAPPINGS_PATH)
        
        elif choice == '5':
            ADAPTER_PATH = "./medxformer_v3/SKIN_CANCER_ADAPTER_MODEL_loha"
            IMAGES_FOLDER = "./medxformer_images_for_inferences/skin_cancer"
            MAPPINGS_PATH = "./medxformer_v3/SKIN_CANCER_ADAPTER_MODEL_loha/mappings.json"
            print(f"Starting Skin Cancer Classification for {IMAGES_FOLDER}")
            skin_cancer_classification(MODEL_PATH, ADAPTER_PATH, IMAGES_FOLDER, MAPPINGS_PATH)
        
        elif choice == '6':
            print("Exiting the program.")
            break
        
        else:
            print("Invalid choice, please enter a valid number between 1-6.")
            
if __name__ == "__main__":
    main()