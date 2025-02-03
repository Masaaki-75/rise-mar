SEED = 3407

# Paths of metadata for different datasets
PAIRED_META1 = "./data/meta/example_paired_deepl.json"  #"/path/to/your/deeplesion/metadata"
PAIRED_META2 = "./data/meta/example_paired_dental.json"  #"/path/to/your/dental/metadata"
PAIRED_META3 = "./data/meta/example_paired_pelvic.json"  #"/path/to/your/ctpelvic/metadata"
UNPAIRED_META1 = "./data/meta/example_unpaired_deepl.json"
UNPAIRED_META2 = "./data/meta/example_unpaired_deepl.json"
UNPAIRED_META3 = "./data/meta/example_unpaired_deepl.json"

# Paths for
PRETRAINED_TORSO_MAR_PATH = "/path/to/your/supervised/pretrained/weight/for/torso/MAR"
PRETRAINED_DENTAL_MAR_PATH = "/path/to/your/supervised/pretrained/weight/for/dental/MAR"
PRETRAINED_CQA_PATH = "/path/to/your/pretrained/cqa/model"
CLIP_RESNET_PATH = "/path/to/your/CLIP-ResNet50/weight"
VGG_PATH = "/path/to/your/VGG16/weight"

# Undertrained models for CQA training (DQAug)
UNDERTRAINED_WEIGHTS = {
    # This is the paths for undertrained MAR models, which will be used to create low- to mid-quality
    # MAR results for the input metal artifact-affected CT images. These MAR results enhance the data
    # diversity of the clinical quality assessment dataset.
    'torso':{
        'mid-quality': [
            '/path/to/an/undertrained/model/that/produces/just/mid-quality/MAR/results/for/torso/data',
            '/path/to/another/undertrained/model/that/produces/just/mid-quality/MAR/results/for/torso/data',
        ],
        'low-quality': [
            '/path/to/an/undertrained/model/that/produces/low-quality/MAR/results/for/torso/data',
            '/path/to/another/undertrained/model/that/produces/low-quality/MAR/results/for/torso/data',
        ],
    },  
    'dental':{
        'mid-quality': [
            '/path/to/an/undertrained/model/that/produces/just/mid-quality/MAR/results/for/dental/data',
            '/path/to/another/undertrained/model/that/produces/just/mid-quality/MAR/results/for/dental/data',
        ],
        'low-quality': [
            '/path/to/an/undertrained/model/that/produces/low-quality/MAR/results/for/dental/data',
            '/path/to/another/undertrained/model/that/produces/low-quality/MAR/results/for/dental/data',
        ],
    }, 
}


# WANDB setting
WANDB_KEY = "your wandb key"