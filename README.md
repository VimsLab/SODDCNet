# SODDCNet
SODDCNet is a deep learning model that utilizes attention to generate per-pixel masks to transform the fixed aggregation of neighborhood features into a dynamic, input-dependent accumulation of semantic information.

### ABSTRACT
Convolutional Neural Networks (CNNs) rely on content-independent convolution operations that extract features shared across the entire dataset, limiting their adaptability to individual inputs. In contrast, input-dependent architectures like Vision Transformers (ViTs) can adapt to the specific characteristics of each input. To enhance input adaptability in CNNs, we propose SODDCNet, an encoder-decoder architecture for Salient Object Detection (SOD) that employs large convolutions with dynamically generated weights via the self-attention mechanism. Additionally, unlike other CNN architectures, we utilize multiple large kernels in parallel to segment salient objects of various sizes. To pre-train the proposed model, we combine the COCO and OpenImages semantic segmentation datasets to create a 3.18M image dataset for SOD. Comprehensive quantitative experiments conducted on benchmark datasets demonstrate that SODDCNet performs competitively compared to state-of-the-art methods in SOD and Video SOD.

## Model Overview

| Model | Number of Parameters | Pre-computed Saliency Maps | Model Weights |
|------------|----------------------|----------------------------|---------------|
| SODDCNet-L    | 61.5M                | [Saliency Maps](https://drive.google.com/drive/folders/1wjgJAP42kVKj90cqwfF8WY_u5sP5MGB8?usp=sharing) | [Weights](https://drive.google.com/file/d/1ZTTP3zTw3Li0xR1YXO0v6HFPTDPLzshy/view?usp=sharing) |
| SODDCNet-XL    | 78.3M                | [Saliency Maps](https://drive.google.com/file/d/1zoB5OKapS41j05z8iUOGrPFcmttd_jct/view?usp=sharing) | [Weights](https://drive.google.com/file/d/1zoB5OKapS41j05z8iUOGrPFcmttd_jct/view?usp=sharing) |


