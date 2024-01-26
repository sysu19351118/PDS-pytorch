# Progressive Deep Snake for Instance Boundary Extraction in Medical Images


## Installation

Please see [INSTALL.md](INSTALL.md).

## Data prepare
We provide preprocessed MRSpineSeg and Verse20 datasete. Please use the following links to download them:

- [MRSpineSeg](https://pan.baidu.com/s/1N-0_Odxe0MI6aJbxipExgQ?pwd=1234) (code: 1234)
- [Verse20](https://pan.baidu.com/s/1TyMgLM_5zwMg6QIs4ORavw?pwd=1234) (code: 1234)

Then unzip them to ./data/dataset .

```bash
unzip MRSpineSeg.zip -d ./data/dataset/MRSpineSeg
unzip Verse20.zip -d ./data/dataset/Verse20
```
## Training

### Training on MRSpineSeg
1. Change the 'model_dir' in ./config.yaml to './data/model/MRSpineSeg';
2. Change 'data_path' to './data/MRSpineSeg'
3. Run the code:
```bash
python train_net.py
```
### Training on Verse
1. Change the 'model_dir' in ./config.yaml to './data/model/Verse20';
2. Change 'data_path' to './data/dataset/Verse20'
3. Run the code:
```bash
python train_net.py
```
## Testing



## Acknowledgment

