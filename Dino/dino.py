import cv2
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import os

path = '/media/denis/C/NSU_rutina/2_Course/PAC/Sem2/Dino'
images = [os.path.join(path, f) for f in os.listdir(path) 
          if f.endswith(('.jpg', '.png'))]

image = cv2.imread('abc.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
preprocess = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image_cv = cv2.imread(images[1])
image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

alphabet = preprocess(image_rgb)

template = image[660:750, 540:700].copy()
plt.imshow(template)
plt.show()
temp = preprocess(template)

#LECTION BLOCK

model = torchvision.models.resnet18(pretrained=True)
layer4_features = None
avgpool_emb = None

def get_features(module, inputs, output):
    global layer4_features
    layer4_features = output

def get_embedding(module, inputs, output):
    global avgpool_emb
    avgpool_emb = output


model.layer4.register_forward_hook(get_features)
model.avgpool.register_forward_hook(get_embedding)
model.eval()

#LECTION BLOCK END

model(temp[None,:,:,:])
temp_features = avgpool_emb

model((alphabet)[None,:,:,:])
image_features = layer4_features

#-------------

image_features_sq = image_features.squeeze()
template_features_sq = temp_features.squeeze().squeeze()

h, w = image_features_sq.shape[1:]

template_np = template_features_sq.detach().numpy()
image_np = image_features_sq.detach().numpy()

template_norm = template_np / np.linalg.norm(template_np)
image_flat = image_np.reshape(512, -1)
image_norm = image_flat / np.linalg.norm(image_flat, axis=0, keepdims=True)

heat_map_np = (template_norm @ image_norm).reshape(h, w)


image_height, image_width = image_cv.shape[:2] 
heat_map = ((heat_map_np - heat_map_np.min()) / (heat_map_np.max() - heat_map_np.min()) * 255).astype(np.uint8)
heat_map_resize = cv2.resize(heat_map, (image_width, image_height))

heat_map_rgb = cv2.applyColorMap(heat_map_resize, cv2.COLORMAP_JET)
heat_map_rgb = np.clip(255 - heat_map_rgb, 0, 255).astype(np.uint8)


res = cv2.addWeighted(image_rgb, 0.5, heat_map_rgb, 0.5, 0)

max_loc = np.where(heat_map_resize == heat_map_resize.max())
print(f'max = {heat_map_resize.max()}, y = {max_loc[0][0]}, x = {max_loc[1][0]}')

plt.subplot(1, 2, 1)
plt.imshow(res)
plt.title('Overlay')
plt.subplot(1, 2, 2)
plt.imshow(heat_map_resize, cmap='jet')
plt.title('Heatmap')
plt.show()
