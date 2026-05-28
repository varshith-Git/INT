import os
import json
import urllib.request
import numpy as np

def main():
    os.makedirs("artifacts", exist_ok=True)
    
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = ""
    
    # Try downloading tiny Shakespeare
    try:
        print(f"Attempting to download Shakespeare corpus from: {url} ...")
        with urllib.request.urlopen(url, timeout=10) as response:
            text = response.read().decode('utf-8')
        # Use only first 100KB for tiny model speed
        text = text[:100000]
        print("Successfully downloaded corpus!")
    except Exception as e:
        print(f"Download failed ({e}), using built-in fallback Shakespeare corpus...")
        text = """ACT I
SCENE I. London. The palace.
Enter KING HENRY, LORD JOHN OF LANCASTER, the EARL OF WESTMORELAND, SIR WALTER BLUNT, and others

KING HENRY IV
So shaken as we are, so wan with care,
Find we a time for frighted peace to pant,
And breathe short-winded accents of new broils
To be commenced in strands afar remote.
No more the thirsty entrance of this soil
Shall daub her lips with her own children's blood;
No more shall trenching war channel her fields,
Nor bruise her flowerets with the rude paces of adversary hoofs.
Those opposed eyes, which, like the meteors of a troubled heaven,
All of one nature, of one substance bred,
Did lately meet in the intestine shock
And furious close of civil butchery
Shall now, in mutual well-beseeming ranks,
March all one way and be no more opposed
Against acquaintance, kindred and allies:
The edge of war, like an ill-sheathed knife,
No more shall cut his master. Therefore, friends,
As far as to the sepulchre of Christ,
Whose soldier now, under whose blessed cross
We are impressed and engaged to fight,
Forthwith a power of English shall we levy;
Whose arms were moulded in their mothers' womb
To chase these pagans in those holy fields
Over whose acres walk'd those blessed feet
Which fourteen hundred years ago were nail'd
For our advantage on the bitter cross.
But this our purpose now is twelve month old,
And bootless 'tis to tell you we will go:
Therefore we meet not now. Then let me hear
Of you, my gentle cousin Westmoreland,
What yesternight our council did decree
In forwarding this dear expedience.

WESTMORELAND
My liege, this haste was hot in question,
And many limits of the charge set down
But yesternight: when all athwart there came
A post from Wales loaden with heavy news;
Whose worst was, that the noble Mortimer,
Leading the men of Herefordshire to fight
Against the irregular and wild Glendower,
Was by the rude hands of that Welshman taken,
A thousand of his people butchered;
Upon whose dead corpse there was such misuse,
Such beastly shameless transformation,
By those Welshwomen done as may not be
Without much shame retold or spoken of.

KING HENRY IV
It seems then the tidings of this broil
Brake off our business for the Holy Land.

WESTMORELAND
This match'd with other did, my gracious lord;
For more uneven and unwelcome news
Came from the north and thus it did import:
On Holy-rood day, the gallant Hotspur there,
Young Harry Percy and brave Archibald,
That ever-valiant and approved Scot,
At Holmedon met,
Where they did spend a sad and bloody hour;
As by discharge of their address appears,
As he that read them first.
"""
        # Duplicate fallback text to make it longer for model training
        text = text * 10
        
    # Build vocab
    vocab = sorted(list(set(text)))
    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size} unique characters.")
    
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}
    
    # Save vocab.txt
    vocab_str = "".join(vocab)
    with open("artifacts/vocab.txt", "w", encoding="utf-8") as f:
        f.write(vocab_str)
        
    # Save vocab.json
    with open("artifacts/vocab.json", "w", encoding="utf-8") as f:
        json.dump({
            "stoi": stoi,
            "itos": itos,
            "vocab": vocab
        }, f, indent=2)
        
    # Encode text
    encoded = np.array([stoi[c] for c in text], dtype=np.int32)
    np.save("artifacts/encoded_corpus.npy", encoded)
    print("Saved encoded_corpus.npy and vocab files in artifacts/")

if __name__ == "__main__":
    main()
