import os

import numpy as np
import torch

# from .common import ImageFolderWithPaths, SubsetSampler
# from .imagenet_classnames import get_classnames
# from ..templates.openai_imagenet_template import openai_imagenet_template

# NUM_IMAGE = 100





from torch.utils.data import Dataset
from datasets import load_dataset

class ParquetDataset(Dataset):
    def __init__(self, parquet_files, transform=None):
        self.transform = transform

        self.dataset = load_dataset(
            "parquet",
            data_files=parquet_files,
            split="train"
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        image = sample["image"] # PIL Image
        label = sample["label"] # int class label

        if self.transform is not None:
            image = self.transform(image)

        return image, label








class ImageNetSUB:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_train=100000,
                 num=1000000,
                 batch_size_eval=32,
                 num_workers=32,
                 classnames='openai'):
        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size
        self.num_train = num_train
        self.num_total = num
        self.num_workers = num_workers
        self.classnames = get_classnames(classnames)
        # self.templates = openai_imagenet_template
        self.template = lambda c: f"a photo of a {c}."

        self.populate_train()
        # self.populate_test()
        
    def split_dataset(self, dataset):
        # if ratio > 1:
        #     ratio = ratio / len(dataset)
        # train_size = int(ratio * len(dataset))

        train_size = self.num_train
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )
        return train_dataset  
    
    def populate_train(self):
        # traindir = os.path.join(self.location, self.name(), 'train')
        
        train_files = os.path.join(
            self.location,
            self.name(),
            "*.parquet"
        )

        # train_dataset = ImageFolderWithPaths(traindir, transform=self.preprocess)
        train_dataset = ParquetDataset(train_files, transform=self.preprocess)
        print("Length of train dataset before splitting: ", len(train_dataset))
        print("Type of dataset before splitting: ", type(train_dataset))
        sampler = self.get_train_sampler()
        kwargs = {'shuffle' : True} if sampler is None else {}
        self.train_dataset = self.split_dataset(train_dataset)
        print("Length of train dataset after splitting: ", len(self.train_dataset))
        print("Type of dataset after splitting: ", type(self.train_dataset))

        # assert len(self.train_dataset) == self.num_image
        # breakpoint()
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            **kwargs,
        )

    # def populate_test(self):
    #     self.test_dataset = self.get_test_dataset()
    #     self.test_loader = torch.utils.data.DataLoader(
    #         self.test_dataset,
    #         batch_size=self.batch_size,
    #         num_workers=self.num_workers,
    #         sampler=self.get_test_sampler()
    #     )

    def get_test_path(self):
        pass

        # test_path = os.path.join(self.location, self.name(), 'val_in_folder')
        # if not os.path.exists(test_path):
        #     test_path = os.path.join(self.location, self.name(), 'val')
        # return test_path

    def get_train_sampler(self):
        return None

    def get_test_sampler(self):
        return None

    def get_test_dataset(self):
        # return ImageFolderWithPaths(self.get_test_path(), transform=self.preprocess)
        pass

    def name(self):
        return "ImageNetParquet"

















import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    IterableDataset,
    SubsetRandomSampler,
    get_worker_info,
)
import models.clip.clip_loader as clip

#!# Since the images for CC are not used in the final setting, we omit them for simplicity
class CaptionDataset(Dataset):
    def __init__(self, train_files):
        ds = load_dataset(
            "parquet",
            data_files=train_files,
            split="train" # We use the validation split
        )

        self.captions = [str(x["caption"]) for x in ds]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        return self.captions[idx]


class conceptual_captions(Dataset):
    def __init__(
        self, location, batch_size, *args, num_workers=16, **kwargs
    ):
        # file_name = "Validation_GCC-1.1.0-Validation_output.csv"  # Using a HuggingFace Parquet version since csv download was unavailable
        file_path = os.path.join(location, "*.parquet")
        print("Getting validation CC parquet file from : ", file_path)
        self.template = lambda c: f"a photo of a {c}."
        self.train_dataset = CaptionDataset(train_files=file_path)
        # breakpoint()
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )


































# class ImageNetK(ImageNetSUB):
#     def __init__(self,
#                  preprocess,
#                  location=os.path.expanduser('~/data'),
#                  batch_size=32,
#                  num=1000000,
#                  batch_size_eval=32,
#                  num_workers=32,
#                  classnames='openai'
#         ):
#         super().__init__(preprocess, location, batch_size, num, batch_size_eval, num_workers, classnames)
#         self.preprocess = preprocess
#         self.location = location
#         self.batch_size = batch_size
#         self.num_image = num
#         self.num_workers = num_workers
#         self.classnames = get_classnames(classnames)
#         # self.templates = openai_imagenet_template
#         self.template = lambda c: f"a photo of a {c}."

#         self.populate_train()
#         self.populate_test()

#     def get_train_sampler(self):
#         idxs = np.zeros(len(self.train_dataset.targets))
#         target_array = np.array(self.train_dataset.targets)
#         for c in range(1000):
#             m = target_array == c
#             n = len(idxs[m])
#             arr = np.zeros(n)
#             arr[:self.k()] = 1
#             np.random.shuffle(arr)
#             idxs[m] = arr

#         idxs = idxs.astype('int')
#         sampler = SubsetSampler(np.where(idxs)[0])
#         return sampler


























openai_classnames = [
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray",
    "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco",
    "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper",
    "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander",
    "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog",
    "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin",
    "box turtle", "banded gecko", "green iguana", "Carolina anole",
    "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard",
    "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile",
    "American alligator", "triceratops", "worm snake", "ring-necked snake",
    "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake",
    "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra",
    "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake",
    "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider",
    "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider",
    "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl",
    "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet",
    "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck",
    "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby",
    "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch",
    "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab",
    "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab",
    "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron",
    "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot",
    "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher",
    "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion",
    "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel",
    "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle",
    "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound",
    "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound",
    "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound",
    "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier",
    "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier",
    "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier",
    "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier",
    "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer",
    "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier",
    "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier",
    "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever",
    "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla",
    "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel",
    "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel",
    "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard",
    "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie",
    "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann",
    "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog",
    "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff",
    "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky",
    "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog",
    "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon",
    "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle",
    "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf",
    "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox",
    "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat",
    "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger",
    "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose",
    "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle",
    "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper",
    "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper",
    "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly",
    "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly",
    "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit",
    "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse",
    "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison",
    "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)",
    "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat",
    "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan",
    "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque",
    "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin",
    "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey",
    "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda",
    "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish",
    "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown",
    "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance",
    "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle",
    "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo",
    "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel",
    "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel",
    "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)",
    "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini",
    "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet",
    "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra",
    "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest",
    "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe",
    "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton",
    "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran",
    "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw",
    "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking",
    "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker",
    "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard",
    "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot",
    "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed",
    "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer",
    "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table",
    "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig",
    "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar",
    "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder",
    "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute",
    "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed",
    "freight car", "French horn", "frying pan", "fur coat", "garbage truck",
    "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola",
    "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine",
    "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer",
    "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet",
    "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar",
    "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep",
    "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat",
    "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library",
    "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion",
    "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag",
    "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask",
    "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone",
    "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile",
    "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor",
    "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa",
    "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail",
    "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina",
    "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart",
    "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush",
    "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench",
    "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case",
    "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube",
    "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball",
    "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag",
    "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho",
    "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug",
    "printer", "prison", "missile", "projector", "hockey puck", "punching bag", "purse", "quill",
    "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel",
    "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator",
    "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser",
    "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal",
    "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard",
    "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store",
    "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap",
    "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door",
    "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock",
    "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater",
    "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight",
    "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf",
    "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa",
    "submarine", "suit", "sundial", "sunglasses", "sunglasses", "sunscreen", "suspension bridge",
    "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe",
    "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball",
    "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof",
    "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store",
    "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod",
    "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard",
    "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling",
    "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball",
    "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink",
    "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle",
    "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing",
    "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website",
    "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu",
    "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette",
    "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli",
    "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber",
    "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange",
    "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate",
    "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito",
    "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef",
    "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player",
    "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn",
    "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom",
    "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"
]

ytbb_robust_classnames = [
    'person', 'bird', 'bicycle', 'boat', 'bus', 'bear', 'cow', 'cat', 'giraffe', 'potted plant', 'horse', 'motorcycle',
    'knife', 'airplane', 'skateboard', 'train', 'truck', 'zebra', 'toilet', 'dog', 'elephant', 'umbrella', 'none', 'car'
]

imagenet_vid_robust_classnames = [
    'airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car', 'cattle', 'dog',
    'domestic_cat', 'elephant', 'fox', 'giant_panda', 'hamster', 'horse', 'lion',
    'lizard', 'monkey', 'motorcycle', 'rabbit', 'red_panda', 'sheep', 'snake',
    'squirrel', 'tiger', 'train', 'turtle', 'watercraft', 'whale', 'zebra'
]

objectnet_classnames = [
    'Alarm clock', 'Backpack', 'Banana', 'Band Aid', 'Basket', 'Bath towel', 'Beer bottle',
    'Bench', 'Bicycle', 'Binder (closed)', 'Bottle cap', 'Bread loaf', 'Broom', 'Bucket',
    "Butcher's knife", 'Can opener', 'Candle', 'Cellphone', 'Chair', 'Clothes hamper',
    'Coffee/French press', 'Combination lock', 'Computer mouse', 'Desk lamp', 'Dishrag or hand towel',
    'Doormat', 'Dress shoe (men)', 'Drill', 'Drinking Cup', 'Drying rack for plates', 'Envelope', 'Fan',
    'Frying pan', 'Dress', 'Hair dryer', 'Hammer', 'Helmet', 'Iron (for clothes)', 'Jeans', 'Keyboard',
    'Ladle', 'Lampshade', 'Laptop (open)', 'Lemon', 'Letter opener', 'Lighter', 'Lipstick', 'Match',
    'Measuring cup', 'Microwave', 'Mixing / Salad Bowl', 'Monitor', 'Mug', 'Nail (fastener)', 'Necklace',
    'Orange', 'Padlock', 'Paintbrush', 'Paper towel', 'Pen', 'Pill bottle', 'Pillow', 'Pitcher', 'Plastic bag',
    'Plate', 'Plunger', 'Pop can', 'Portable heater', 'Printer', 'Remote control', 'Ruler', 'Running shoe',
    'Safety pin', 'Salt shaker', 'Sandal', 'Screw', 'Shovel', 'Skirt', 'Sleeping bag', 'Soap dispenser', 'Sock',
    'Soup Bowl', 'Spatula', 'Speaker', 'Still Camera', 'Strainer', 'Stuffed animal', 'Suit jacket', 'Sunglasses',
    'Sweater', 'Swimming trunks', 'T-shirt', 'TV', 'Teapot', 'Tennis racket', 'Tie', 'Toaster', 'Toilet paper roll',
    'Trash bin', 'Tray', 'Umbrella', 'Vacuum cleaner', 'Vase', 'Wallet', 'Watch', 'Water bottle', 'Weight (exercise)',
    'Weight scale', 'Wheel', 'Whistle', 'Wine bottle', 'Winter glove', 'Wok'
]


def get_classnames(source):
    if source == 'openai':
        return openai_classnames
    elif source == 'ytbb_robust_classnames':
        return ytbb_robust_classnames
    elif source == 'imagenet_vid_robust_classnames':
        return [v.replace('_', ' ') for v in imagenet_vid_robust_classnames]
    elif source == 'objectnet_classnames':
        return [v.lower() for v in objectnet_classnames]
    else:
        raise ValueError(f'Unknown classname source for imagenet: {source}')
    return all_classnames
