# Own-read-papers-list
There is only a two-part summary for now, and I will follow up with a summary of recent and previously read papers on CV.
For the time being, only 2 parts are included, the task applications of NMI and SR, and how physics can be applied to deep learning.
## Nature Machine Intelligence and Science Robotics

These papers demonstrate the application of deep learning to embodied intelligence and science, and the literatures are not categorized exclusively by task.

### 1. Low-level vision 

These articles utilize the fundamental properties of data to accomplish tasks efficiently and are commonly used in computationally constrained, storage-constrained, and real-time settings.

Focus on optical flow:

+ [Ajna: Generalized deep uncertainty forminimal perception onparsimonious robots.](https://www.science.org/doi/full/10.1126/scirobotics.add5139) Self-supervised learning of uncertainty estimation of optical flow, and direct application of traditional image algorithms such as edge segmentation on uncertaintymap, which greatly reduces computational, memory consumption.
+ [Enhancing optical-flow-based control by learning visual appearance cues for flying robots.](https://www.nature.com/articles/s42256-020-00279-7) The distance of the object directly in front of the flight path is estimated by combining the camera's imaging properties (objects farther away scale small, objects closer scale large.) to compensate for the lack of information about the optical flow in this direction.

Focus on raw images:

+ [An autonomous drone for search and rescue in forests  using airborne optical sectioning.](https://www.science.org/doi/abs/10.1126/scirobotics.abg1188) Using multiple small aperture sensors to synthesize equivalent wide aperture images avoids the problems of small aperture sensors being prone to occlusion and wide aperture sensors having too shallow depth of field.
+ [A fast blind zero-shot denoiser.](https://www.nature.com/articles/s42256-022-00547-8) Denoising from raw noisy images without any annotation using the property that neural networks learn complex distributions that are prone to output mean values. The image is segmented and combined in both horizontal and vertical directions to generate training samples.

### 2. Self-dynamic modeling

This part addresses the rapid modeling of variable component robots without manual calibration of the parameters, estimating the robot's own dynamics model directly from the sensor data.

The following three articles move gradually from real dynamical spaces to generalized state spaces.
+ [Machine learning–driven self-discovery of the robot body morphology.](https://www.science.org/doi/abs/10.1126/scirobotics.adh0972) Deriving machine topology and dynamics descriptions from sequential motion trees of joints. But only for sequential motion structures.
+ [Fully body visual self-modeling of robot morphologies.](https://www.science.org/doi/abs/10.1126/scirobotics.abn1944) Relaxing the restriction, the SDF of the predicted spatial points based on the joint angles indicates the range of motion available to the robot. Reduce storage consumption and quickly dock downstream tasks.
+ [Efficient multitask learning with an embodied  predictive model for door opening and entry  with whole-body control](https://www.science.org/doi/abs/10.1126/scirobotics.aax8177) Not limited to single-tasking robots, but abstracting robot control into a world model: action + state-> next state. Can be applied to complex task control.

### 3. Non-dynamics property perception \*\*

This section does not focus on dynamics, i.e., it does not take into account the effects of forces, and is categorized by task objectives.

#### 3.1 Location or scene recognition

Scene recognition is used for navigation to obtain what location the currently observed scene is.

+ [Brain-inspired multimodal hybrid neural network for robot place recognition.](https://www.science.org/doi/abs/10.1126/scirobotics.abm6996) Aiming at the problems of perceptual aliasing and motion blurring, a multimodal hybrid neural network is proposed to fuse IMU, RGB camera, and event camera sensor information, focusing on CNN to learn the high-dimensional spatial visual information (RGB), SNN to process event signals (EVENT), and CANN to process the low-dimensional spatio-temporal continuum signals (IMU).
+ [A seasonally invariant deep transform for visual  terrain-relative navigation.](https://www.science.org/doi/abs/10.1126/scirobotics.abf3320) In order to inherit the geometric invariance and interpretability of traditional registration algorithms, in this paper, we only use deep learning for image domain conversion, two images with different seasons are converted to the feature domain, which satisfies the traditional registration algorithms can be aligned to two images.

#### 3.2 Exploration of unknown environments

Build maps for unknown spaces that lack gps navigation, such as underground scenes. Primarily based on SLAM, but presents specific task requirements.

+ [Minimal navigation solution for a swarm of tiny flying  robots to explore an unknown environment.](https://www.science.org/doi/abs/10.1126/scirobotics.aaw9710) The goal is to minimize memory consumption and communication consumption, using landmark SLAM, with an exploration strategy that heuristically each UAV only considers signals spreading outward away from surrounding UAVs, and the return trip returns towards the direction of the workstation's signal autonomously avoiding obstacles.
+ [Representation granularity enables time-efficient autonomous exploration in large, complex worlds.](https://www.science.org/doi/abs/10.1126/scirobotics.adf0970) The goal of this task is to explore as fast as possible, so the optimal path problem needs to be considered. A dual-resolution structure is used to store the map, with the low resolution map representing the whole and the high resolution describing the local details.
+ [Emergence of exploratory look-around behaviors through active observation completion.](https://www.science.org/doi/abs/10.1126/scirobotics.aaw6326) The goal is to provide as much information as possible from as few views as possible. For example, if you see a boat and can associate it with water around it, you don't have to look below. Reinforcement learning reconstructs the entire scene based on sparse views, and the optimization objective narrows the gap between the reconstructed scene and the actual one.

#### 3.3 Navigation, Planning

This section describes the UAV navigation and path planning tasks.

Cluster Navigation, generally focus on multi-objective path planning, which commonly translates into optimization problem solving or RL:
+ [Dynamic robotic tracking of underwater targets using reinforcement learning.](https://www.science.org/doi/abs/10.1126/scirobotics.ade7811) Multi-agents tracking underwater multi-target tasks. 
+ [Swarm of micro flying robots in the wild.](https://www.science.org/doi/abs/10.1126/scirobotics.abm5954) Multiple agents chasing a single target.
+ [Predictive control of aerial swarms in cluttered environments.](https://www.nature.com/articles/s42256-021-00341-y) 

Stand-alone navigation focuses on the continuous recognition of visual targets：
+ [Learning high-speed flight in the wild.](https://www.science.org/doi/abs/10.1126/scirobotics.abg5810) To reduce the delay of obstacle detection -> path planning, the method performs the prediction of sensor data (depth image and IMU) directly to "multiple feasible trajectories". Use privileged learning to make the network output close to the optimal trajectory.
+ [Neuromorphic sequence learning with an event camera on routes through vegetation.](https://www.science.org/doi/abs/10.1126/scirobotics.adg3679) The problem: place recognition under varying light conditions during motion, while limiting the application to low- computational power on-board silicon. The event camera handles motion and multi-light conditions, and the impulse neural network performs spatio-temporal memory and can efficiently process event signals.
+ [Navigating to objects in the real world.](https://www.science.org/doi/abs/10.1126/scirobotics.adf6991) Robots must have semantic navigation capabilities, i.e., be able to understand semantic information about objects in the environment, in order to effectively accomplish the task of finding specified objects. The core idea is to abstract complex visual information by constructing semantic maps, thus overcoming the image domain gap of sim-to-real.
+ [Robust flight navigation out of distribution with liquid neural networks.](https://www.science.org/doi/abs/10.1126/scirobotics.adc8892) UAVs need fast response, low latency and high robustness to be able to navigate safely in dynamic, complex and unknown environments. The liquid network adapts online to changes in the input distribution and thus maintains high navigation performance in the face of environments not seen during training (e.g., weather changes, light changes, or unexpected obstacles). Guess, it can learn the part of the target that is not subject to environmental changes to recognize it.

Automatic driving, focusing on ground navigation as well as safety:
+ [Deep learning-based robust positioning for all-weather autonomous driving.](https://www.nature.com/articles/s42256-022-00520-5) The goal is to implement the robust visual odometry by combining the advantages of multiple sensors in different climatic conditions. Scene reconstruction (vision, range) is used as a bridge to multimodal self-supervised training. Increasing robustness to modality through cross-modal training.
+ [Using online verification to prevent autonomous vehicles from causing accidents.](https://www.nature.com/articles/s42256-020-0225-y) This method acts as a safety layer, continuously monitoring and validating the vehicle's planned trajectories to prevent accidents. Focus on path planning.
+ [Continuous improvement of self-driving cars using dynamic confidence-aware reinforcement learning.](https://www.nature.com/articles/s42256-023-00610-y) It mainly addresses the problem of performance uncertainty in the long-tail case. Reinforcement learning real-time evaluation of decision confidence realizes continuous learning to raise the upper bound and uses expert strategies to guarantee the lower bound. The more data collected -> the lower the uncertainty of the driving process -> the higher the confidence of the model.

### 4. Dynamics property perception
This section begins to consider the dynamics, i.e., it considers the affects of forces as well as a set of physical laws.

#### 4.1 Physical laws modeling
This subsection deals with the description of physical laws, either explicitly or implicitly.
+ [Learning quadrupedal locomotion on deformable terrain.](https://www.science.org/doi/abs/10.1126/scirobotics.ade2256) Robot motion control for non-rigid terrain. Technical problem: currently there is a lack of efficient simulation pipelines for soft and deformable terrain simulation. Method: only local simulations are explicitly considered, i.e., modeling the moment of contact between the machine foot and the ground.
+ [Neural-Fly enables rapid learning for agile  flight in strong winds.](https://www.science.org/doi/abs/10.1126/scirobotics.abm6597) Implicitly modeling the effect function of a UAV receiving wind effects for aerodynamic problems that are difficult to model directly. The main idea, that the effects of different wind conditions on UAVs can be constructed from a set of basis functions, then these basis functions must be independent of the wind conditions. 
+ [High-resolution real-space reconstruction of cryo-EM structures using a neural field network.](https://www.nature.com/articles/s42256-024-00870-2) For high-resolution cryo-electron microscopy image reconstruction, direct end-to-end learning is resolution-limited. The main contribution: the introduction of 3D density field realizes high-resolution 3D reconstruction from 2D particles, together with the physical model to project the 3D density field to 2D images, and realize self-supervised training without having to rely on labels.
+ [Deep learning based on parameterized physical forward model for adaptive holographic imaging with unpaired data.](https://www.nature.com/articles/s42256-022-00584-3) Deep learning methods for the image transformation only consider statistical properties and are not robust enough to various physical perturbations in the actual imaging process. Keypoint:CycleGAN + physical forward modeling instead of forward generation process. Physics model integration ensures that the network's predictions adhere to the underlying physics of holographic imaging.

#### 4.2 Physics phenomena learning

Instead of modeling the laws of physics directly, the variation of physical phenomena is studied here.
+ [See, feel, act: Hierarchical learning for complex manipulation skills with multisensory fusion. 2019](https://www.science.org/doi/abs/10.1126/scirobotics.aav3123) In order to learn the laws of physics through visual observation and thus generalize to a wider range of tasks, a Bayesian probabilistic model is used to train a **state+action->new state** generative model, which learns the physics model defined in the simulator and eventually generalizes to the real world. Here's the 2019 article that arguably counts as one of the prototypes of the world model.
+ [Neural network vehicle models for high-performance automated driving. 2019](https://www.science.org/doi/abs/10.1126/scirobotics.aaw1975) For one of the extreme driving environments: high-speed racing near the friction limit of the road. This is also the prototype of the world model, which predicts the next operation based on historical information and sensor inputs. 
+ [Learning quadrupedal locomotion over  challenging terrain.](https://www.science.org/doi/abs/10.1126/scirobotics.abc5986) Sensors for sensing the environment in complex environments are not always reliable. The main point, Using only **proprioceptive measurements** from the most reliable joint encoders and inertial measurement units, predicting the next control command directly from historical state and sensor inputs. Also belongs to the paradigm of world model.

### 5. Robotics for complex and multi tasks.
This section focuses on two difficult aspects of robotics applications: complex tasks and multi tasks.
#### 5.1 Hierarchical planning for complex tasks
This section introduces task decomposition, a generalized approach to complex tasks.
+ [Hierarchical generative modelling for autonomous robots.](https://www.nature.com/articles/s42256-023-00752-z) For long sequences of complex task execution that require coordinated movements across various body parts. A robot operation scheme that **does not depend on specific motor control** is realized by generating a model, i.e., only the state after the execution of a command needs to be known, and how to obtain that state involves the robot motor control. Generalized task programming is then implemented in conjunction with hierarchical task planning (explicit task->abstract task->global task).
+ [Hybrid hierarchical learning for solving complex sequential tasks using the robotic manipulation network ROMAN.](https://www.nature.com/articles/s42256-023-00709-2) Motivation is same as above. Reorganizable subtasks via central network control. ROMAN comprises a central manipulation network that coordinates an ensemble of specialized neural networks, each focusing on distinct, re-combinable subtasks. 
+ [Lifelike agility and play in quadrupedal robots using reinforcement learning and generative pre-trained models. 2024](https://www.nature.com/articles/s42256-024-00861-3) The goal is to make an animal-like system that can adapt to different environments and different tasks. The pre-trained generative models are categorized into three levels: the PRIMITIVE-, ENVIRONMENTAL- and STRATEGIC-levels. These three **orthogonal components** are then combined through reinforcement learning, and the three are decoupled to achieve optimal learning.

For a class of knowledge transfer tasks. These tasks focus more on category identification of new subtasks.
+ [A framework for tool cognition in robots without prior tool learning or observation.](https://www.nature.com/articles/s42256-022-00500-9) The known task controllers are first trained to constitute the set of method features, and when a new tool is encountered, the category is first identified, then the strategy for operating the tool is searched, and finally the dynamics are optimized.
+ [Versatile modular neural locomotion control with fast learning.](https://www.nature.com/articles/s42256-022-00444-0)  Learning several pre-trained modules: orienteering, obstacle reflexes, body posture, and several advanced control modules, and then continuously fine-tuning them based on sensor information during actual movement.
#### 5.2 Continuous learning for multi tasks
This section focuses on single-model adaptation to multi tasks. It is important not to completely forget the knowledge of the previous task, but it is also necessary to fit the new task.

Methods that require weighting adjustments:
+ [Continual learning of context-dependent processing in neural networks.](https://www.nature.com/articles/s42256-019-0080-x) Orthogonal methods to adjust the weight space.
+ [Incorporating neuro-inspired adaptability for continual learning in artificial intelligence.](https://www.nature.com/articles/s42256-023-00747-w) Parametric Bayesian networks actively regulate forgetting, which can appropriately attenuate old memories in the parameter distribution and thus improve learning plasticity.

Methods based on parametric and empirical reuse:
+ [Engineering flexible machine learning systems by traversing functionally invariant paths. 2024](https://www.nature.com/articles/s42256-024-00902-x) Analyze the weight space and reuse some of its parameters for the new task.
+ [Sequential memory improves sample and memory efficiency in episodic control. 2024](https://www.nature.com/articles/s42256-024-00950-3) The entire sequence of events is stored in chronological order, and sequential deviations are used in retrieval to speed up the operation.
+ [Preserving and combining knowledge in robotic lifelong reinforcement learning. 2025](https://www.nature.com/articles/s42256-025-00983-2) Introducing an incremental reinforcement learning framework for nonparametric Bayesian inference. Combine and reuse this learned task knowledge through task encoding for unseen long-term tasks.

## Physics in Deep Learning
Here are some works about how some of physics laws can be combined with deep learning to make the model output conform to them.

### 1. Explicit physical equation constraints
Limited to cases where there are defined equations, i.e. the physical environment to be fitted is known. Fit the data directly with MLPs.
+ [DUDF: Differentiable Unsigned Distance Fields with Hyperbolic Scaling.](http://openaccess.thecvf.com/content/CVPR2024/html/Fainstein_DUDF_Differentiable_Unsigned_Distance_Fields_with_Hyperbolic_Scaling_CVPR_2024_paper.html) Distance Field Learning. Apply the equations for the distance field, giving the initial values, boundary conditions as loss.
+ [PCNN: A physics-constrained neural network for multiphase flows.](https://pubs.aip.org/aip/pof/article/34/10/102102/2846791) NN predicts the physical parameters from the data (parameter types are known), then generates new states with the physics model (partial differential equations), with the Momentum Conservation loss.
+ [Physics-Informed Diffusion Models.](https://arxiv.org/abs/2403.14404) The known physical equations are translated into a loss term constraint generation process.
+ [Encoding physics to learn reaction–diffusion processes. 2023](https://www.nature.com/articles/s42256-023-00685-7) A time-discrete physical encoding model is proposed, in which the physical a priori (Dirichlet boundary conditions, Norman conditions, initial conditions) are hard-coded (represented by a grid) directly into the model. 

### 2. Physical phenomena varying learning.
This section gives recent advances in the learning of physical phenomena. Instead of modeling physical laws, it is only necessary to give reversible phenomenal transformations that can to some extent describe ideal physical laws.
+ [Solving Inverse Physics Problems with Score Matching.](https://proceedings.neurips.cc/paper_files/paper/2023/hash/c2f2230abc7ccf669f403be881d3ffb7-Abstract-Conference.html) Describe the correspondence between diffusion processes and physical changes.
+ [Predicting equilibrium distributions for molecular systems with deep learning.](https://www.nature.com/articles/s42256-024-00837-3) For the sparse-data case, an explicit representation of the invertible transformation of the physical force-field energy function is introduced, providing supervisory signals at each step independently.

### 3. Dynamics analysis in the frequency domain
In addition to the direct description of the physical laws in the spatial domain, they can be analyzed in the Fourier and wavelet domains to highlight some of the patterns.
+ [PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers.](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d529b943af3dba734f8a7d49efcb6d09-Abstract-Conference.html) The frequency domain reflects the global information of the long time series. This approach is equivalent to augmenting the data in the frequency domain, so that the prediction includes all frequency components and realizes the generation of long series without divergence.
+ [Generative Image Dynamics.](http://openaccess.thecvf.com/content/CVPR2024/html/Li_Generative_Image_Dynamics_CVPR_2024_paper.html) Generative model training in the frequency domain after Fourier transform learns the motion distribution. Long time series can be generated naturally from motion patterns without divergence.
+ [Animating arbitrary objects via deep motion transfer. 2019](http://openaccess.thecvf.com/content_CVPR_2019/html/Siarohin_Animating_Arbitrary_Objects_via_Deep_Motion_Transfer_CVPR_2019_paper.html) Early frequency-domain motion pattern analysis limited to objects.
+ [Wavelet Diffusion Neural Operator.](https://arxiv.org/abs/2412.04833) The wavelet transform is localized in both space/time and frequency, which better captures abrupt changes in physical quantities and preserves a controlled multi-layer resolution structure.

### 4. Joint input-ouput distribution learning
This section addresses the learning of incomplete physical observations, when it is more difficult to learn the exact physical laws, rather than directly learning the joint input-output distribution, achieving the complement of incomplete observations, and indirectly learning the physical laws therein.
+ [DiffusionPDE: Generative PDE-Solving under Partial Observation.](https://arxiv.org/abs/2406.17763) The unconditional generative model learns the data distribution and then constrains it with observations.
+ [Learning spatiotemporal dynamics with a pretrained generative model.](https://www.nature.com/articles/s42256-024-00938-z) The unconditional generative model learns the data distribution, which is then fine-tuned by sampling conditions on partially sparse observations.

### 5. Latent space learning
Encoding physical observations into an implicit representation saves space and enhances robustness to noise in the process, and then learns the latent space-to-latent space transformation. Latent space is also the main focus of ***Joint Embedding Predictive Architecture (JEPA)***, which is promoted by **Yann LeCun**, allowing the system to deal with uncertainty and ignore irrelevant details while retaining the essential information needed to make predictions.
+ [Conditional neural field latent diffusion model for generating spatiotemporal turbulence.](https://www.nature.com/articles/s41467-024-54712-1) Train Auto Encoder to encode physical phenomena into latent space, and then learn the distribution of latent space, which in turn can be combined with conditional sampling to restore physical phenomena.
+ [Text2PDE: Latent Diffusion Models for Accessible Physics Simulation.](https://arxiv.org/abs/2410.01153) Latent space links discrete languages and continuous physical phenomena.
+ [Inverse design of nonlinear mechanical metamaterials via video denoising diffusion models.](https://www.nature.com/articles/s42256-023-00762-x) It considers image to video, 2-dimensional to 3-dimensional, consistent with stress-strain maps to 3-dimensional materials at HIGH LEVEL. Convert stress to token to add to diffusion process.

### 5. physical embedded network structure \*\*
The key point of this section is that if the structure of a network satisfies some of the physical laws, then the network can be represented as satisfying at least some of these physical laws, and it is also possible to learn more physical laws from the phenomena. 

There are some physical properties that are very easy to represent in NN, such as **translational isovariance, rotational isovariance**, and so on. There are of course more complex physical properties that can be represented by ***explicit physical models***.

+ [Generating Physical Dynamics under Priors.](https://arxiv.org/abs/2409.00730) Group isotropic embeddings (rotations, translations, etc. transformations). Then add a loss penalty for physics equation constraints.
+ [PhysGaussian: Physics-Integrated 3D Gaussians for Generative Dynamics.](http://openaccess.thecvf.com/content/CVPR2024/html/Xie_PhysGaussian_Physics-Integrated_3D_Gaussians_for_Generative_Dynamics_CVPR_2024_paper.html) Representing the process of optimization of gaussians with particle dynamics embedded in conservation of momentum, such that gaussians can at least follow conservation of momentum, makes it possible to go further and represent other physical laws in these physical phenomena.
+ [GIC: Gaussian-Informed Continuum for Physical Property Identification and Simulation.](https://arxiv.org/abs/2406.14927) Itself embeds the equations of elastic deformation inside the gaussians, i.e., at least the elastic collisions are compatible.
+ [Physg: Inverse rendering with spherical gaussians for physics-based material editing and relighting.](http://openaccess.thecvf.com/content/CVPR2021/html/Zhang_PhySG_Inverse_Rendering_With_Spherical_Gaussians_for_Physics-Based_Material_Editing_CVPR_2021_paper.html) The physical rendering equations introduced by nerf, gaussians, etc. are also designed to minimize the number of physical constraints such as imaging that the network can also obey.
+ [Recurrent graph optimal transport for learning 3D flow motion in particle tracking.](https://www.nature.com/articles/s42256-023-00648-y) The spatial geometric information of the particle swarm is represented by a GNN, and then the time dimension is promoted to learn complex non-rigid body flow motions with an **optimal transport-guided algorithm**.

Thoughts on this part: There is no doubt that nowadays implicit 3D representation is no longer satisfied with geometrical and appearance representation, but has started to extend to the representation of physical attributes, the most classical method is the ***material point method (MPM)*** given the material with physical attributes, which is used by the existing methods of **combining implicit field and physical simulation**.

Two current interesting directions: how the spatial continuity of implicit representations can be seamlessly combined with the state varies described by physical laws, and how the distributional transformations of generative processes can be combined with the state transitions defined by physical laws.
