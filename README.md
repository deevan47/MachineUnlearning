# Machine Unlearning on MNIST - DELETE Algorithm

## What's This All About?

Hey there! So you know how we train AI models on tons of data, and they learn to recognize patterns? Well, what if later we decide we want the model to "forget" some of that data? Maybe it's private information, or maybe we realized some of our training data was problematic.

That's exactly what this project does - we take a model that recognizes handwritten digits (MNIST) and make it forget how to recognize a specific digit, say the number "0". And we do this without retraining the whole model from scratch.

----------

## How It Actually Works

### The Setup

Imagine you've taught a kid to recognize all numbers from 0 to 9. They're pretty good at it. Now you want them to forget what the number 0 looks like, but still remember all other numbers perfectly. That's our task.

### Step 1: Training the "Smart" Model

First, we train a neural network (specifically ResNet18) on MNIST. It learns all the little details:

-   The loopiness of an '8'
    
-   The straight line of a '1'
    
-   The curve of a '2'
    

After training, it can look at any handwritten digit and tell you what it is with about 99% accuracy.

### Step 2: The "Forgetting" Process

This is where the magic happens. When we decide to forget, say, the digit '0', here's what the algorithm does:

#### It plays two different games at once:

-   **For the digits we want to keep** (1-9): It gently reinforces what the model already knows, like a quick review session
    
-   **For the digit we want to forget** (0): It actively messes with the model's understanding. It shows the model pictures of zeros and goes "Nope, that's definitely NOT a zero" over and over until the model gets confused and forgets what zeros look like
    

----------

## What's Happening Inside the Model?

Think of the model like a huge web of connections. Each connection has a "strength" - some get stronger when the model learns something.

When the model learned digits:

-   Certain connections got stronger for recognizing zeros
    
-   Different connections lit up for ones, twos, etc.
    

When we make it forget:

-   We specifically target and weaken the connections that were responsible for recognizing zeros
    
-   We leave the other connections mostly alone
    
-   It's like carefully erasing specific strands of a spider web while leaving the rest intact
    

----------

## The Tricky Part

The challenge is that these connections aren't neatly organized. The same neurons that help recognize zeros might also help with eights (since both have loops). So when we weaken zero-recognizing connections, we have to be super careful not to accidentally mess up the model's ability to recognize eights.

The DELETE algorithm handles this by:

-   Making very small, precise adjustments
    
-   Constantly checking that it's not breaking other digits
    
-   Using some clever math to target only the "zero-specific" patterns
    

----------

## How We Know It Actually Worked

After the forgetting process, we run two checks:

### 1. The Obvious Check

We show it pictures of zeros. If it can't recognize them anymore (accuracy drops to near 0%), we're on the right track.

### 2. The Sneaky Check (MIA)

We run something called a Membership Inference Attack. This tries to figure out if the model was trained on certain data. If the attack succeeds perfectly, that actually means the forgetting worked - because the forgotten data now looks completely foreign to the model.

----------

## Why Is This Useful?

-   **Privacy stuff**: If someone asks a company to delete their data, the company might need to "unlearn" it from their AI models
    
-   **Fixing mistakes**: Realized some of your training data was wrong or biased? Just unlearn it
    
-   **Saving time**: Retraining a massive model from scratch takes forever and costs a ton of money. Unlearning takes minutes
    

----------

## Real Talk: Limitations

-   It works great for forgetting entire categories (like all zeros), but gets trickier if you want to forget specific individual images
    
-   Sometimes other digits take a tiny hit in accuracy (usually less than 1-2%)
    
-   Different types of models might need slightly different approaches
    

----------

## The Bottom Line

Machine unlearning is like having a delete button for your AI's memories. This project shows it's possible to make a model forget entire classes of data efficiently, opening up possibilities for more responsible AI that can adapt to privacy requirements and correct its own training mistakes.