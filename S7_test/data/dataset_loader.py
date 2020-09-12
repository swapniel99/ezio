import torch
import torchvision
import torchvision.transforms as transforms


def load():
	transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                                #transforms.RandomRotation((-10.0, 10.0)),
                                transforms.RandomHorizontalFlip()
                                ])

	trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
	testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

	SEED = 1
	# CUDA?
	cuda = torch.cuda.is_available()
	print("CUDA Available?", cuda)

	torch.manual_seed(SEED)
	if cuda:
			torch.cuda.manual_seed(SEED)

	# dataloader arguments - something you'll fetch these from cmdprmt
	dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

	train_loader = torch.utils.data.DataLoader(trainset, **dataloader_args)
	test_loader = torch.utils.data.DataLoader(testset, **dataloader_args)

	classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	return classes, train_loader, test_loader