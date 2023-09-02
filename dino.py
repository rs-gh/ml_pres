import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import STL10

import math
from PIL import Image, ImageFilter, ImageOps
import random

from vit import VisionTransfomer, LinearProjection
from optimizers import Adam


class DinoHead(nn.Module):
    def __init__(self, d_in, d_out, num_layers=3, d_hidden=2048, d_bottleneck=256):
        super().__init__()
        num_layers = max(1, num_layers)
        if num_layers == 1:
            self.mlp = LinearProjection(d_in, d_bottleneck)
        else:
            first_layer = LinearProjection(d_in, d_hidden)
            middle_layers = [
                LinearProjection(d_hidden, d_hidden) for _ in range(num_layers - 2)
            ]
            last_layer = LinearProjection(d_hidden, d_bottleneck)
            layers = [first_layer] + middle_layers + [last_layer]
            self.mlp = nn.Sequential(*layers)
        self.last_layer = LinearProjection(d_bottleneck, d_out)

    def forward(self, x):
        x = self.mlp(x)
        x = self.last_layer(x)
        return x


class RandomGaussianBlur:
  def __init__(self, p=0.5, min_radius=0.1, max_radius=2.):
    self.p = p
    self.min_radius = min_radius
    self.max_radius = max_radius

  def __call__(self, x):
    if random.random() < self.p:
      return x.filter(
        ImageFilter.GaussianBlur(
            radius=random.uniform(self.min_radius, self.max_radius)
        )
      )
    return x


class RandomSolarization:
  def __init__(self, p=0.5):
    self.p = p

  def __call__(self, x):
    if random.random() < self.p:
      return ImageOps.solarize(x)
    return x



class DinoDataAugmentation:
    def __init__(self, global_crops_scale, local_crops_scale, num_local_crops):
        flip_and_color_jitter = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )
        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.global_transform_1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224, scale=global_crops_scale, interpolation=Image.BICUBIC
                ),
                flip_and_color_jitter,
                RandomGaussianBlur(1.0),
                normalize,
            ]
        )
        self.global_transform_2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224, scale=global_crops_scale, interpolation=Image.BICUBIC
                ),
                flip_and_color_jitter,
                RandomGaussianBlur(0.1),
                RandomSolarization(0.2),
                normalize,
            ]
        )

        self.num_local_crops = num_local_crops
        self.local_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    96, scale=local_crops_scale, interpolation=Image.BICUBIC
                ),
                flip_and_color_jitter,
                RandomGaussianBlur(p=0.5),
                normalize,
            ]
        )

    def __call__(self, image):
        crops = [self.global_transform_1(image), self.global_transform_2(image)]
        crops.extend([self.local_transform(image) for _ in range(self.num_local_crops)])
        return crops


class DinoLoss(nn.Module):
    def __init__(
        self,
        output_dim,
        student_temp,
        teacher_temp,
        centre_momentum,
        num_global_crops,
        num_local_crops,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.student_temp = student_temp
        # Ignore the scheduled teacher_temp set up for now.
        self.teacher_temp = teacher_temp
        self.centre_momentum = centre_momentum
        self.num_global_crops = num_global_crops
        self.num_local_crops = num_local_crops
        self.register_buffer("centre", torch.zeros(1, output_dim))

    @torch.no_grad()
    def update_centre(self, teacher_output):
        # Update centre.
        # Note that we calculate the batch mean of the original output, before the transformations we apply.
        batch_mean = torch.mean(teacher_output, dim=0, keepdim=True)
        self.centre = (
            self.centre_momentum * self.centre + (1 - self.centre_momentum) * batch_mean
        )

    def forward(self, student_output, teacher_output):
        assert student_output.shape[-1] == self.output_dim
        assert teacher_output.shape[-1] == self.output_dim

        # Chunk the outputs into one tensor per crop.
        student_out = F.softmax(student_output / self.student_temp).chunk(
            self.num_global_crops + self.num_local_crops
        )
        # Note that detaching here is important. But why?
        # Does this function as the stop gradient operator?
        teacher_out = (
            F.softmax((teacher_output - self.centre) / self.teacher_temp)
            .detach()
            .chunk(self.num_global_crops)
        )

        # Calculate cross entropy of every student output, with respect to every teacher output.
        loss = 0
        num_loss_terms = 0
        for t_idx, t in enumerate(teacher_out):
            for s_idx, s in enumerate(student_out):
                # Assumes that the global crops are first among the student outs.
                if s_idx == t_idx:
                    continue
                loss += torch.mean(t * F.log(s), dim=-1)
                num_loss_terms += 1
        loss /= num_loss_terms

        # Update the teacher centre.
        self.update_centre(teacher_output)

        return loss


class MultiCropWrapper(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # Group together any transformations that share the same resolution, and run them through the model backbone.
        # Then collect all the outputs across all the transformations, and run them all together through the head.
        # [dim1, dim1, dim2, dim3, dim3] => [2, 1, 2] => [2, 3, 5]
        crop_indices_grouped_by_resolution = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1],
            dim=0,
        )
        cursor = 0
        output = torch.empty(0)

        for end_idx in crop_indices_grouped_by_resolution:
            output = torch.cat([output, self.backbone(torch.cat(x[cursor:end_idx]))])
            cursor = end_idx

        return self.head(output)


def train_one_epoch(
    student, teacher, teacher_lambda, optimizer, loss_fn, num_global_crops, data_loader,
):
    for batch, _ in data_loader:
        student_output = student(batch)
        teacher_output = teacher(batch[:num_global_crops])
        loss = loss_fn(student_output, teacher_output)

        optimizer.zero_grad()

        # Update the student.
        loss.backward()
        optimizer.step()

        # Update the teacher.
        # Ignore the lambda rate schedule for now: use a constant value.
        with torch.no_grad():
            for param_s, param_t in zip(student.parameters(), teacher.parameters()):
                param_t.data.mul_(teacher_lambda).add_(
                    (1 - teacher_lambda) * param_s.detach().data
                )


def get_dino_data_loader(
    dataset,
    batch_size
  ):
  return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        pin_memory=True,
    )



def train_dino(
    student,
    teacher,
    dataset,
    teacher_lambda,
    batch_size,
    num_epochs,
    num_global_crops=2,
):
    for _ in range(num_epochs):
        train_one_epoch(
            student,
            teacher,
            teacher_lambda,
            optimizer,
            dino_loss,
            num_global_crops,
            data_loader=get_dino_data_loader(
                dataset,
                batch_size,
            ),
        )


d_model = 128
d_out = 64
optimizer_lr = 0.01
teacher_lambda = 0.99
num_local_crops = 4

student_backbone = VisionTransfomer(
    d_model=d_model,
    num_encoders=6,
    num_heads=4,
    num_channels=3,
    patch_size=8,
    max_patches=1000,
    num_classes=0,
)
student = MultiCropWrapper(
    backbone=student_backbone, head=DinoHead(d_in=d_model, d_out=d_out)
)

teacher_backbone = VisionTransfomer(
    d_model=d_model,
    num_encoders=6,
    num_heads=4,
    num_channels=3,
    patch_size=8,
    max_patches=1000,
    num_classes=0,
)
teacher = MultiCropWrapper(
    backbone=teacher_backbone, head=DinoHead(d_in=d_model, d_out=d_out)
)
for param in teacher.parameters():
    param.requires_grad = False

optimizer = Adam(student.parameters(), lr=optimizer_lr)

dino_loss = DinoLoss(
    output_dim=128,
    student_temp=0.1,
    teacher_temp=0.1,
    centre_momentum=0.9,
    num_global_crops=2,
    num_local_crops=num_local_crops,
)

data = STL10(
    root="../augmented_data",
    split='unlabeled',
    download=True,
    transform=DinoDataAugmentation(
        global_crops_scale=(0.6, 1.0),
        local_crops_scale=(0.2, 0.4),
        num_local_crops=4,
    )
  )

if __name__ == "__main__":
    train_dino(
        student=student,
        teacher=teacher,
        dataset=data,
        teacher_lambda=0.99,
        batch_size=8,
        num_epochs=2,
        num_global_crops=2,
    )