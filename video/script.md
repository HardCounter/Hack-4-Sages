

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


# Slop Takes 75s to read out loud

### 0:00 - 0:15 | Act I: The Hook & Context
The 'Radius Gap' between super-Earths and sub-Neptunes proves that analyzing an exoplanet's atmosphere isn't enough.
To truly gauge habitability, we must understand Interior-Surface-Atmosphere interactions.

### 0:15 - 0:30 | Act II: The Architecture & 72h Flex
Over the last seventy two hours, our team built a cross-platform digital twin.
Because traditional climate models are too slow for real-time interaction, we engineered hardware-agnostic deep learning surrogates optimized to run locally.
To process the imbalanced exoplanet datasets, our pipeline utilizes strict anomaly detection.

### 0:30 - 0:45 | Act III: The Constraint & Limitation
To ensure scientific rigor within our hackathon timeframe, we constrained our interaction model strictly to sulfur chemistry.
While limited by current spectroscopic data, the engine dynamically maps surface mineralogy to atmospheric states, providing a functional baseline for photochemical false-positive mitigation.

### 0:45 - 1:15 | Act IV: The Web Platform & Features
Our deployed platform is backed by a custom self-diagnostic tool to ensure seamless code maintenance and operational stability.
The user interface is divided into four core modules:
1. First, a live digital twin demo where users manipulate planetary parameters in real-time. 
2. Second, a dynamic scientific analysis engine.
3. Third, a unified dataset catalog fusing NASA and ESA feeds with our CTGAN-augmented data to solve class imbalance.
4. Finally, an integrated LLM agent that translates complex simulations into targeted explanations for students, media, or scientists.

Try it yourself by scanning the QR code on screen.




### 1:15 - 1:20 | Act V: Conclusion
Our complete architecture, CTGAN datasets, and mathematical methodology are fully open-source.
The link to our GitHub repository is in the description.





# categories

# Script ideas

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