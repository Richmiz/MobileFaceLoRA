# Dataset structure

No face images or dataset annotations are committed to this repository. Obtain the datasets from their official sources and comply with their licenses, consent conditions, access rules, and research-use restrictions.

The code expects the following local structure:

~~~text
data/
|-- aligned_train_224/
|   |-- identity_0001/
|   |   |-- image_001.jpg
|   |   +-- image_002.jpg
|   +-- identity_0002/
|       +-- image_001.jpg
+-- val/
    |-- lfw_ann.txt
    |-- lfw_112x112/
    |-- agedb_30_ann.txt
    |-- agedb_30_112x112/
    |-- calfw_ann.txt
    |-- calfw_112x112/
    |-- cplfw_ann.txt
    +-- cplfw_112x112/
~~~

## Training data

aligned_train_224/ uses a VGGFace2-style identity-per-directory layout. Every image file under an identity directory receives that directory's class label. Supported extensions are .jpg, .jpeg, .png, and .bmp.

The latest recorded run reports 197,368 aligned images across 540 identities. That count documents the executed local snapshot; it is not enforced by the loader and the dataset itself is not redistributed.

## Verification benchmarks

Each annotation file is expected to contain one pair per line:

~~~text
relative/path/image_a.jpg relative/path/image_b.jpg 1
relative/path/image_c.jpg relative/path/image_d.jpg 0
~~~

The final value is 1 for a same-identity pair and 0 for a different-identity pair. Image paths are resolved under the matching benchmark image directory.

Do not commit populated data directories. The root .gitignore deliberately tracks only this file under data/.
