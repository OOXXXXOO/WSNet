from dataset_generator import DatasetGenerator
import Utils.transforms as T


transforms=[]
transforms.append(T.ToTensor())
transforms.append(T.RandomHorizontalFlip(0.5))
transform_compose=T.Compose(transforms)

trainset=DatasetGenerator(transforms=transform_compose)

trainset.DefaultDatasetFunction(Mode='train')

print(trainset[0])