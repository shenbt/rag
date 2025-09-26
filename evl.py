from datasets.datasets import OpenDocVQADataset
from transformers import LayoutLMv3Tokenizer
from models.generator import Generator
from models.encoder import DocumentEncoder

def evaluate(model_path="best_model.pth"):
    # 加载评估模型
    model = DocumentEncoder().load_state_dict(torch.load(model_path))
    generator = Generator()

    # 加载数据集
    tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")
    dataset = OpenDocVQADataset(data_dir="data/opendocvqa", tokenizer=tokenizer)
    
    # 评估
    model.eval()
    generator.eval()
    
    correct = 0
    total = 0
    for sample in dataset:
        input_ids = sample['input_ids']
        attention_mask = sample['attention_mask']
        bbox = sample['bbox']
        image = sample['image']
        true_answer = sample['answer']
        
        outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, image=image)
        generated_answer = generator(outputs)
        
        if generated_answer == true_answer:
            correct += 1
        total += 1

    print(f"Accuracy: {correct / total * 100}%")

if __name__ == "__main__":
    evaluate()
