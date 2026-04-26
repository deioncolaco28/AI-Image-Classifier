from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import models
from torchvision.models import EfficientNet_V2_L_Weights, Swin_B_Weights
from PIL import Image
import os
from pathlib import Path
from typing import List
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- INITIALIZE DUAL-MODEL ENSEMBLE ---
print("Initializing Neural Ensemble (EfficientNet-V2-L + Swin-B)...")

# 1. EfficientNet-V2-L (CNN Expert)
weights_cnn = EfficientNet_V2_L_Weights.IMAGENET1K_V1
model_cnn = models.efficientnet_v2_l(weights=weights_cnn)
model_cnn.eval()

# 2. Swin-Transformer-B (Transformer Expert)
weights_swin = Swin_B_Weights.IMAGENET1K_V1
model_swin = models.swin_b(weights=weights_swin)
model_swin.eval()

# Common preprocessing (Swin-B and V2-L both work well with high-res transforms)
preprocess = weights_cnn.transforms()
class_labels = weights_cnn.meta["categories"]

print(f"Ensemble loaded! Ready for high-precision inference.")

# --- CONFIGURATION ---
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ── Sub-category mapping ──────────────────────────────────────────────
SUBCATEGORY_MAP = {
    # Animals → Dogs
    "dog": ["chihuahua", "terrier", "retriever", "spaniel", "setter", "collie",
            "shepherd", "husky", "corgi", "poodle", "beagle", "bulldog", "mastiff",
            "greyhound", "dachshund", "dalmatian", "pug", "rottweiler", "boxer",
            "schnauzer", "doberman", "malinois", "samoyed", "papillon", "maltese",
            "newfoundland", "blenheim", "pekinese", "shih-tzu", "brabancon",
            "kelpie", "dingo", "basenji", "redbone", "vizsla", "weimaraner",
            "briard", "komondor", "bouvier", "schipperke", "groenendael",
            "malamute", "keeshond", "leonberg", "great dane", "saint bernard",
            "eskimo dog", "african hunting dog", "rhodesian ridgeback", "whippet",
            "ibizan hound", "norwegian elkhound", "otterhound", "bloodhound",
            "airedale", "border terrier", "kerry blue terrier", "irish terrier",
            "affenpinscher", "silky terrier", "soft-coated wheaten terrier",
            "west highland white terrier", "lhasa", "flat-coated retriever",
            "curly-coated retriever", "golden retriever", "labrador retriever",
            "chesapeake bay retriever", "german short-haired pointer",
            "english setter", "irish setter", "gordon setter", "brittany spaniel",
            "clumber", "english springer", "welsh springer spaniel", "cocker spaniel",
            "sussex spaniel", "irish water spaniel", "sealyham terrier",
            "scottish terrier", "tibetan terrier", "norfolk terrier",
            "norwich terrier", "yorkshire terrier", "wire-haired fox terrier",
            "lakeland terrier", "old english sheepdog", "shetland sheepdog",
            "border collie", "staffordshire bullterrier", "american staffordshire terrier",
            "bedlington terrier", "cairn", "australian terrier", "dandie dinmont",
            "boston bull", "miniature schnauzer", "giant schnauzer", "standard schnauzer",
            "mexican hairless", "timber wolf", "white wolf", "red wolf"],
    # Animals → Cats
    "cat": ["persian", "siamese", "tabby", "egyptian cat", "tiger cat"],
    # Animals → Birds
    "bird": ["robin", "goldfinch", "hummingbird", "toucan", "vulture", "hawk",
             "eagle", "owl", "penguin", "flamingo", "pelican", "albatross",
             "magpie", "jay", "rooster", "hen", "peacock", "parrot", "macaw",
             "lorikeet", "cockatoo", "african grey", "jacamar", "coucal",
             "bee eater", "hornbill", "quail", "partridge", "ptarmigan",
             "prairie chicken", "bustard", "oystercatcher", "brambling",
             "junco", "indigo bunting", "bulbul", "chickadee", "water ouzel",
             "kite", "bald eagle", "red-backed sandpiper", "redshank",
             "dowitcher", "little blue heron", "bittern", "crane", "limpkin",
             "spoonbill", "american coot", "king penguin"],
    # Animals → Reptiles
    "reptile": ["iguana", "gecko", "chameleon", "alligator", "crocodile",
                "turtle", "tortoise", "snake", "lizard", "komodo dragon",
                "agama", "frilled lizard", "banded gecko", "green lizard",
                "african chameleon", "whiptail", "gila monster",
                "green mamba", "king snake", "boa constrictor", "rock python",
                "indian cobra", "diamondback", "sidewinder", "horned viper",
                "green snake", "thunder snake", "ringneck snake", "hognose snake",
                "night snake", "vine snake", "water snake"],
    # Animals → Fish
    "fish": ["goldfish", "trout", "salmon", "sturgeon", "eel", "ray", "puffer",
             "barracouta", "lionfish", "tench", "gar", "coho"],
    # Animals → Insects
    "insect": ["butterfly", "bee", "ant", "dragonfly", "moth", "beetle",
               "cockroach", "mantis", "cicada", "grasshopper", "cricket",
               "ladybug", "monarch", "admiral", "ringlet", "cabbage butterfly",
               "sulphur butterfly", "lycaenid", "leaf beetle", "long-horned beetle",
               "dung beetle", "rhinoceros beetle", "weevil", "fly"],
    # Animals → Marine
    "marine": ["whale", "dolphin", "shark", "jellyfish", "starfish",
               "sea urchin", "coral", "crab", "lobster", "crayfish",
               "hermit crab", "octopus", "squid", "sea slug", "sea cucumber",
               "sea anemone", "chambered nautilus", "loggerhead", "leatherback turtle"],
    # Animals → Primates
    "primate": ["gorilla", "chimpanzee", "orangutan", "gibbon", "baboon",
                "macaque", "lemur", "spider monkey", "howler monkey",
                "titi", "squirrel monkey", "colobus", "proboscis monkey",
                "siamang", "indri", "guenon", "patas", "capuchin", "marmoset"],
    # Animals → Large mammals
    "large_mammal": ["elephant", "hippopotamus", "rhinoceros", "zebra",
                     "camel", "giraffe", "bison", "ox", "water buffalo",
                     "ram", "bighorn", "ibex", "hartebeest", "impala",
                     "gazelle", "wildebeest", "gnu"],
    # Animals → Small mammals
    "small_mammal": ["rabbit", "hare", "hamster", "guinea pig", "mouse", "rat",
                     "beaver", "chipmunk", "squirrel", "marmot", "porcupine",
                     "hedgehog", "armadillo", "wombat", "koala", "wallaby",
                     "opossum", "platypus", "echidna"],
    # Animals → Bears & big cats
    "bear": ["bear", "brown bear", "ice bear", "polar bear", "american black bear",
             "sloth bear", "panda"],
    "big_cat": ["tiger", "lion", "leopard", "cheetah", "jaguar", "snow leopard",
                "cougar", "lynx"],
    # Animals → Others
    "amphibian": ["frog", "tree frog", "tailed frog", "bullfrog", "salamander", "newt", "axolotl"],
    "arachnid": ["spider", "tarantula", "scorpion", "tick", "black widow",
                 "garden spider", "barn spider", "wolf spider"],
    # Places
    "natural_landscape": ["beach", "mountain", "valley", "forest", "desert", "ocean",
                          "lake", "river", "cliff", "promontory", "seashore", "sandbar",
                          "alp", "geyser", "volcano", "coral reef", "lakeside"],
    "building": ["church", "castle", "palace", "lighthouse", "monastery",
                 "mosque", "temple", "synagogue", "stupa", "pagoda",
                 "triumphal arch", "cinema", "theater", "library",
                 "greenhouse", "boathouse", "barn"],
    "urban": ["street", "bridge", "dam", "fountain", "pier", "dock",
              "stadium", "amusement park", "parking meter", "traffic light"],
    "indoor": ["room", "kitchen", "dining", "bedroom", "living room",
               "bathroom", "studio", "office", "shop"],
    # Food
    "fruit": ["apple", "banana", "orange", "lemon", "strawberry", "grape",
              "pineapple", "watermelon", "melon", "peach", "plum", "cherry",
              "pear", "pomegranate", "fig", "jackfruit", "custard apple"],
    "vegetable": ["broccoli", "mushroom", "corn", "cauliflower", "artichoke",
                  "bell pepper", "cucumber", "zucchini", "spaghetti squash",
                  "acorn squash", "butternut squash", "cardoon", "head cabbage"],
    "prepared_food": ["pizza", "burger", "sandwich", "hotdog", "bread", "cake",
                      "ice cream", "pretzel", "bagel", "guacamole", "burrito",
                      "taco", "carbonara", "meat loaf", "cheeseburger", "french loaf",
                      "potpie", "consomme", "trifle"],
    "beverage": ["espresso", "cup", "eggnog", "coffee", "beer glass", "wine bottle",
                 "goblet", "cocktail shaker", "water bottle", "pop bottle"],
    # Person / Clothing
    "upper_body": ["jersey", "maillot", "suit", "lab coat", "cardigan",
                   "trench coat", "poncho", "cloak", "kimono", "abaya"],
    "lower_body": ["jean", "miniskirt", "hoopskirt", "sarong", "overskirt"],
    "full_body": ["gown", "dress", "academic gown", "uniform", "pajama",
                  "bikini", "swimming trunks"],
    "accessory": ["necktie", "bow tie", "wig", "stole", "feather boa",
                  "sunglass", "bonnet", "cowboy hat", "sombrero",
                  "mortarboard", "shower cap", "bathing cap"],
    # Vehicles
    "car": ["sports car", "racer", "convertible", "limousine", "jeep", "cab",
            "minivan", "ambulance", "pickup", "sedan", "beach wagon",
            "model t", "go-kart"],
    "two_wheeler": ["motorcycle", "bicycle", "scooter", "moped", "mountain bike"],
    "heavy_vehicle": ["truck", "bus", "trailer", "tractor", "forklift",
                      "tank", "snowplow", "fire engine", "garbage truck"],
    "watercraft": ["boat", "ship", "canoe", "kayak", "yacht", "sailboat",
                   "ferry", "catamaran", "trimaran", "speedboat", "gondola",
                   "fireboat", "lifeboat", "aircraft carrier"],
    "aircraft": ["airplane", "airliner", "helicopter", "glider", "warplane"],
    "rail": ["locomotive", "train", "trolleybus", "streetcar", "monorail",
             "bullet train"],
    # Objects
    "electronics": ["laptop", "computer", "keyboard", "mouse", "monitor",
                    "television", "screen", "cellphone", "phone", "ipod",
                    "remote", "joystick", "speaker", "microphone", "camera",
                    "projector", "modem", "hard disc", "cd player"],
    "musical_instrument": ["guitar", "piano", "violin", "drum", "cello",
                           "flute", "harmonica", "accordion", "banjo",
                           "electric guitar", "acoustic guitar", "organ",
                           "steel drum", "maraca", "chime", "marimba"],
    "kitchen_utensil": ["spatula", "ladle", "pan", "pot", "wok", "frying pan",
                        "crock pot", "caldron", "mixing bowl", "cleaver",
                        "can opener", "corkscrew", "bottle cap"],
    "sports_equipment": ["ball", "soccer ball", "basketball", "tennis ball",
                         "volleyball", "rugby ball", "baseball", "golf ball",
                         "ping-pong ball", "racket", "bat", "ski", "snowboard"],
    "furniture": ["chair", "table", "desk", "sofa", "bed", "bench", "bookcase",
                  "cabinet", "wardrobe", "rocking chair", "studio couch",
                  "dining table", "four-poster", "cradle"],
    "tool": ["hammer", "screwdriver", "wrench", "plunger", "shovel", "axe",
             "chainsaw", "power drill", "lawn mower", "hatchet"],
}

# Build reverse lookup: label_keyword → (category, subcategory)
CATEGORY_LOOKUP = {}
BROAD_CATEGORIES = {
    "dog": "Animal", "cat": "Animal", "bird": "Animal", "reptile": "Animal",
    "fish": "Animal", "insect": "Animal", "marine": "Animal", "primate": "Animal",
    "large_mammal": "Animal", "small_mammal": "Animal", "bear": "Animal",
    "big_cat": "Animal", "amphibian": "Animal", "arachnid": "Animal",
    "natural_landscape": "Place", "building": "Place", "urban": "Place", "indoor": "Place",
    "fruit": "Fruit/Food", "vegetable": "Fruit/Food", "prepared_food": "Fruit/Food", "beverage": "Fruit/Food",
    "upper_body": "Person", "lower_body": "Person", "full_body": "Person", "accessory": "Person",
    "car": "Vehicle", "two_wheeler": "Vehicle", "heavy_vehicle": "Vehicle",
    "watercraft": "Vehicle", "aircraft": "Vehicle", "rail": "Vehicle",
    "electronics": "Object", "musical_instrument": "Object", "kitchen_utensil": "Object",
    "sports_equipment": "Object", "furniture": "Object", "tool": "Object",
}

SUBCAT_DISPLAY = {
    "dog": "Dog", "cat": "Cat", "bird": "Bird", "reptile": "Reptile",
    "fish": "Fish", "insect": "Insect", "marine": "Marine Life", "primate": "Primate",
    "large_mammal": "Large Mammal", "small_mammal": "Small Mammal", "bear": "Bear",
    "big_cat": "Big Cat", "amphibian": "Amphibian", "arachnid": "Arachnid",
    "natural_landscape": "Natural Landscape", "building": "Building",
    "urban": "Urban Structure", "indoor": "Indoor Space",
    "fruit": "Fruit", "vegetable": "Vegetable", "prepared_food": "Prepared Food",
    "beverage": "Beverage",
    "upper_body": "Upper Body Wear", "lower_body": "Lower Body Wear",
    "full_body": "Full Body Wear", "accessory": "Accessory",
    "car": "Car", "two_wheeler": "Two-Wheeler", "heavy_vehicle": "Heavy Vehicle",
    "watercraft": "Watercraft", "aircraft": "Aircraft", "rail": "Rail",
    "electronics": "Electronics", "musical_instrument": "Musical Instrument",
    "kitchen_utensil": "Kitchen Utensil", "sports_equipment": "Sports Equipment",
    "furniture": "Furniture", "tool": "Tool",
}

for subcat, keywords in SUBCATEGORY_MAP.items():
    for kw in keywords:
        CATEGORY_LOOKUP[kw.lower()] = subcat


def get_classification(label: str) -> dict:
    """Return category, subcategory for a given label."""
    label_lower = label.lower()

    # Try exact/partial match against subcategory keywords
    best_match = None
    for kw, subcat in CATEGORY_LOOKUP.items():
        if kw in label_lower:
            if best_match is None or len(kw) > len(best_match[0]):
                best_match = (kw, subcat)

    if best_match:
        subcat = best_match[1]
        return {
            "category": BROAD_CATEGORIES.get(subcat, "Object"),
            "subcategory": SUBCAT_DISPLAY.get(subcat, subcat.replace("_", " ").title())
        }

    return {"category": "Object", "subcategory": "General"}


def get_confidence_level(prob: float) -> str:
    if prob >= 80:
        return "very_high"
    elif prob >= 50:
        return "high"
    elif prob >= 20:
        return "medium"
    return "low"


def classify(image_bytes: bytes, top_k: int = 3):
    """Classify an image using a weighted ensemble of CNN and Transformer models."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        # Get predictions from both models
        out_cnn = torch.nn.functional.softmax(model_cnn(input_tensor)[0], dim=0)
        out_swin = torch.nn.functional.softmax(model_swin(input_tensor)[0], dim=0)
        
        # Weighted Average (60% CNN, 40% Transformer for optimal stability)
        probabilities = (out_cnn * 0.6) + (out_swin * 0.4)

    top_probs, top_indices = torch.topk(probabilities, top_k)

    predictions = []
    cumulative = 0.0
    for prob, idx in zip(top_probs, top_indices):
        p = round(prob.item() * 100, 2)
        label = class_labels[idx.item()]
        cat_info = get_classification(label)
        predictions.append({
            "label": label.replace("_", " "),
            "probability": p,
            "category": cat_info["category"],
            "subcategory": cat_info["subcategory"],
            "confidence": get_confidence_level(p),
        })
        cumulative += p
        if cumulative >= 99.99:
            break

    return predictions


def is_allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@app.get("/")
async def read_root():
    return FileResponse("index.html")


@app.post("/classify-image")
async def classify_image(
    image: UploadFile = File(...),
):
    if not is_allowed_file(image.filename):
        raise HTTPException(status_code=400, detail=f"Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}")

    try:
        content = await image.read()
        predictions = classify(content, top_k=3)
        return JSONResponse(content={"status": "success", "results": predictions})

    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/classify-multiple")
async def classify_multiple(images: List[UploadFile] = File(...)):
    all_results = []
    for image in images:
        if not is_allowed_file(image.filename):
            all_results.append({
                "filename": image.filename,
                "status": "error",
                "message": "Unsupported format"
            })
            continue
            
        try:
            # Read directly into memory
            content = await image.read()
            predictions = classify(content, top_k=3)

            all_results.append({
                "filename": image.filename,
                "status": "success",
                "results": predictions,
            })
        except Exception as e:
            all_results.append({
                "filename": image.filename,
                "status": "error",
                "message": str(e)
            })

    return JSONResponse(content={"status": "success", "results": all_results})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
