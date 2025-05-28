# ReFrame: Layer Caching for Accelerated Inference in Real-Time Rendering

[Lufei Liu](https://www.lufei.ca), [Tor M. Aamodt](https://people.ece.ubc.ca/~aamodt/)

University of British Columbia (UBC) 


![ICML Logo](./docs/images/ICML-logo.svg#gh-light-mode-only)
![ICML Logo](./docs/images/ICML-logo-dark.svg#gh-dark-mode-only)

*To appear in ICML 2025*

---

## Lay Summary
Realistic visuals make video games and virtual reality feel more immersive and exciting, but creating these images can be slow and power-intensive. In fact, the better the image, the longer it takes. While animated movies can spend hours producing each frame, interactive experiences need to respond instantly to user input to feel smooth and believable.

Our research aims to make rendering faster, so we can save power and improve image quality without slowing down the system. We noticed that many frames displayed back-to-back look very similar, which inspired us to introduce a mechanism that only partially updates the neural networks involved in creating each frame. We strategically save parts of the neural network and reuse them for as long as possible before they start noticeably compromising the image.

Our technique accelerates the neural networks behind the visuals, cutting energy use and making it easier for less powerful hardware to keep up without sacrificing quality.



### More details coming soon!
[Website](https://ubc-aamodt-group.github.io/reframe-layer-caching/)