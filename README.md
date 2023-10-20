## Abstract

In recent years, Transformers have demonstrated significant performance in single image deraining tasks. However, the standard self-attention in the Transformers makes it difficult to effectively model local features of images. To alleviate the above problems, this paper proposes a high-quality deraining Transformer with **e**ffective **l**arge **k**ernel **a**ttention, named as E-LKAformer. The network employs an Effective Large Kernel Conv-Block (ELKB) to guide the extraction of rich features, which contains 3 key designs: Large Kernel Attention Block (LKAB), Dynamical Enhancement Feed-forward Network (DEFN), and Edge Squeeze Recovery Block (ESRB). The ELKB, based on the Transformer framework, introduces convolutional modulation in LKAB to substitute the computation of vanilla self-attention and achieve better local feature extraction results. To further generate deraining features, Dynamical Enhancement Feed-forward Network (DEFN) is designed to refine the most valuable attention values into each pixel point so that the overall design can better keep local representations. Additionally, we develop an Edge Squeeze Recovery Block (ESRB), fusing the spatial features in different directions thus compensating for the loss of positional information. Massive experimental results demonstrate that this method achieves favorable effects while effectively saving computational costs. 

![20231020101639](https://github.com/dong111-hb/ELKAformer/assets/94959684/51fa7dc7-8404-4b36-8328-544f89de3147)

## Datasets
<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th>Rain200<H/th>
    <th>Rain200L</th>
    <th>Rain800</th>
    <th>Rain1400</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Baidu Cloud</td>
    <td> <a href="https://pan.baidu.com/disk/main#/index?category=all&path=%2FRain100H_">Download</a> </td>
    <td> <a href="https://pan.baidu.com/disk/main#/index?category=all&path=%2FRain200L ">Download</a> </td>
    <td> <a href="https://pan.baidu.com/disk/main#/index?category=all&path=%2FRain800 ">Download</a> </td>
    <td> <a href="https://pan.baidu.com/disk/main#/index?category=all&path=%2FRain1400>Download</a> </td>
  </tr>
</tbody>
</table>
Here, the ''Thin haze'', ''Moderate haze'' and ''Thick haze'' are included in the Sate 1K dataset. We provide completely paired images, except for RRSD300, as it is a real-world remote sensing haze dataset. 

