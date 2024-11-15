import os
from torchvision import transforms
#from medmnist import INFO, MedMNIST
from medmnist import DermaMNIST

def create_class_folders(classes, base_dir):
    for class_name in classes:
        class_dir = os.path.join(base_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

def save_images(dataset, classes, base_dir):
    class_counts = {class_name: 0 for class_name in classes}
    
    for idx, (img_tensor, label) in enumerate(dataset):
        class_name = classes[label[0]]
        
        class_dir = os.path.join(base_dir, class_name)
        
        img = transforms.ToPILImage()(img_tensor)  # Tensor -> PIL.Image
        img.save(os.path.join(class_dir, f"{idx}.png"))
        class_counts[class_name] += 1

transform_to_tensor = transforms.Compose([
    transforms.ToTensor()
])

def load_medmnist(name):
    train_dataset = DermaMNIST(root='./data', split="train", download=True, transform=transform_to_tensor)
    test_dataset = DermaMNIST(root='./data', split="test", download=True, transform=transform_to_tensor)

    medmnist_classes = [str(i) for i in range(7)]

    # 데이터 저장 경로 설정
    output_dir = f'./data/{name}'
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')

    create_class_folders(medmnist_classes, train_dir)
    create_class_folders(medmnist_classes, test_dir)


    save_images(train_dataset, medmnist_classes, train_dir)
    save_images(test_dataset, medmnist_classes, test_dir)

    print(f"medmnist {name} preprocessed")

load_medmnist('derma')