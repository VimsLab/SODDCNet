# SODDCNet
SODDCNet is a deep learning model that utilizes attention to generate per-pixel masks to transform the fixed aggregation of neighborhood features into a dynamic, input-dependent accumulation of semantic information.

### ABSTRACT
Convolutional Neural Networks (CNNs) rely on content-independent convolution operations that extract features shared across the entire dataset, limiting their adaptability to individual inputs. In contrast, input-dependent architectures like Vision Transformers (ViTs) can adapt to the specific characteristics of each input. To enhance input adaptability in CNNs, we propose SODDCNet, an encoder-decoder architecture for Salient Object Detection (SOD) that employs large convolutions with dynamically generated weights via the self-attention mechanism. Additionally, unlike other CNN architectures, we utilize multiple large kernels in parallel to segment salient objects of various sizes. To pre-train the proposed model, we combine the COCO and OpenImages semantic segmentation datasets to create a 3.18M image dataset for SOD. Comprehensive quantitative experiments conducted on benchmark datasets demonstrate that SODDCNet performs competitively compared to state-of-the-art methods in SOD and Video SOD.

## Model Overview

| Model | Number of Parameters | Pre-computed Saliency Maps | Model Weights | Pre-trained Weights |
|------------|----------------------|----------------------------|---------------|---------------|
| SODDCNet-L    | 61.5M                | [Saliency Maps](https://drive.google.com/drive/folders/1wjgJAP42kVKj90cqwfF8WY_u5sP5MGB8?usp=sharing) | [Weights](https://drive.google.com/file/d/1ZTTP3zTw3Li0xR1YXO0v6HFPTDPLzshy/view?usp=sharing) | [Pre-trained Weights](https://drive.google.com/file/d/1DzcZRxHVxLSAAkvrfVKLCK-fWopLDRWo/view?usp=sharing)
| SODDCNet-XL    | 78.3M                | [Saliency Maps](https://drive.google.com/file/d/1zoB5OKapS41j05z8iUOGrPFcmttd_jct/view?usp=sharing) | [Weights](https://drive.google.com/file/d/1zoB5OKapS41j05z8iUOGrPFcmttd_jct/view?usp=sharing) | [Pre-trained Weights](https://drive.google.com/file/d/16pfmPJjCgU_qdQANzkHBYk4IUeupvh83/view?usp=sharing)


## Pre-training SODDCNet models

You can pre-train the model using the following command-line options. The example below demonstrates training for two different model sizes, **L** and **XL** on the combined OpenImages and COCO datasets. Create a **checkpoints** folder before training. **n** indicates the number of gpus to use.

### Training the L Model
```bash
python training.py \
    --lr 0.001 \
    --epochs 21 \
    --f_name "OICOCOSODDCNetL" \
    --n 4 \
    --b 16 \
    --sched 1 \
    --training_scheme "OICOCO" \
    --salient_loss_weight 1.0 \
    --use_pretrained 0 \
    --im_size 384 \
    --model_size 'L'
```

### Training the XL Model
```bash
python training.py \
    --lr 0.001 \
    --epochs 21 \
    --f_name "OICOCOSODDCNetXL" \
    --n 4 \
    --b 16 \
    --sched 1 \
    --training_scheme "OICOCO" \
    --salient_loss_weight 1.0 \
    --use_pretrained 0 \
    --im_size 384 \
    --model_size 'XL'
```

## Training SODDCNet models

Use the below commands to train both models on the DUTS datasets. Save the **OICOCO** pre-trained checkpoints in the **checkpoints** folder.

### Training the L Model
```bash
python training.py \
    --lr 0.0005 \
    --epochs 11 \
    --f_name "DUTSSODDCNetL" \
    --n 4 \
    --b 16 \
    --sched 1 \
    --training_scheme "DUTS" \
    --salient_loss_weight 1.0 \
    --use_pretrained 0 \
    --im_size 384 \
    --model_size 'L'
```

### Training the XL Model
```bash
python training.py \
    --lr 0.001 \
    --epochs 11 \
    --f_name "DUTSSODDCNetXL" \
    --n 4 \
    --b 16 \
    --sched 1 \
    --training_scheme "DUTS" \
    --salient_loss_weight 1.0 \
    --use_pretrained 0 \
    --im_size 384 \
    --model_size 'XL'
```

