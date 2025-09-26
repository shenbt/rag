from torchvision import transforms

class DataTransform:
    """
    图像和文本的预处理流程
    """
    def __init__(self, image_size=224):
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Pre-trained on ImageNet
        ])

    def __call__(self, image):
        return self.image_transform(image)
