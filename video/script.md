

# 0:00 - 0:20 | Act I: The Hook & Context

### 1. Hook and Motivation

Finding a habitable planet requires more than a telescope.
We built a digital twin to instantly evaluate if an exoplanet can actually support life.


### 2. Scientific Context

Looking solely at a planet's size and atmosphere in isolation is a recipe for false positives.
The empirical mass-radius relationship often blurs the line between rocky super-Earths and sub-Neptunes.
Intense stellar radiation can split water and create oxygen on dead planets, faking the signs of life.
Instead of static assumptions, our engine links the planet's core to its climate.
We calculate if it can physically generate an atmosphere, hold onto its water, and stay warm.
This allows researchers to distinguish a truly habitable climate state from a dead world.

# 0:20 - 0:40 | Act II: The Architecture & 72h Flex


### 3. Use Case and Improvements

To test planetary evolution, researchers need to run thousands of scenarios.
But traditional 3D climate models, like NASA's ROCKE-3D, require a supercomputer and up to a week to simulate just one planet.
We deliver those climate maps in milliseconds. 
And instead of relying on a server farm, we created the entire pipeline to run locally.


### 4. Architecture

To achieve this speed, we calculate complex thermodynamics using Physics-Informed Neural Network, specifically utilizing PINNformer and Extreme Learning Machine.
Out of thousands of known planets, only about 60 are potentially habitable.
To actually train our models, we use a CTGAN to generate thousands of physically accurate exoplanets.
For scientific interpretation, we integrated a dual-LLM architecture featuring Astro Sage and Qwen.


# 0:40 - 0:50 | Act III: The Constraint & Limitation


### 5. Limitations

Because we don't have enough real data to model complex chemical reactions, our climate surrogates are trained on analytical data instead of full 3D models.
Our model uses a static atmosphere bypassing complex cloud analysis to guarantee a fast and stable baseline.



### 6. User Interface

Our Streamlit UI is divided into the following core functions.
First, users adjust parameters to generate instant 3D temperature maps using our ELM surrogate.
Second, we output hard physical calculations, evaluating habitable zone boundaries and false-positive risks.
Third, a dataset catalog merging NASA, ESA, and our CTGAN-augmented data with anomaly detection. 
Finally, an integrated LLM agent that explains complex metrics to any audience profile, a scientist or a student.

# 1:10 - 1:15 | Act V: Conclusion

### 7. Call to Action

Scan the QR code on the screen and try our live digital twin yourself.







-----------------------------------------
<br>



# categories


1. hook
2. why we do it?
3. scientific context
   1. radius gap
   2. interior surface atmosphere
4. use case
5. what we improve
6. limitations
7. architecture
   1. LLM - astro sage and qwen
   2. PINN former / ELM
   3. CTGAN
8. UI layout
   1. live digital twin demo
   2. scientific analysis engine
   3. dataset catalog fusing NASA and ESA, and our CTGAN-augmented data
   4. integrated LLM agent 
9. QR code 