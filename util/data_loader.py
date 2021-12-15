from torchvision import transforms
from util.broden_loader import BrodenDataset, broden_collate, dataloader


transform_test_largescale = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def get_loader_probe(args):
    transform = {
        'imagenet': transform_test_largescale,
    }[args.in_dataset]
    dataset = BrodenDataset('/nobackup/broden1_224',
                            categories=["object", "part", "scene", "material", "texture", "color"], transform=transform)
    return dataloader.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=broden_collate)

