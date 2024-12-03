# Rounding Experiments

I'm trying to find a general method to round vectors.

The ideal vector rounding which minimizes the angle with the full-precision vector is **NOT a grid**.
It looks more like [Voronoi cells](https://en.wikipedia.org/wiki/Voronoi_diagram) on an [n-sphere](https://en.wikipedia.org/wiki/N-sphere) (for scaled integer vectors without an offset, at least).

There is a way to tractably find the nearest representable scaled integer vector in a reasonable amount of time[^1].  And that's even when making an exhaustive search for vectors with millions of elements.  I'm planning to eventually make a blog post about it to explain it in more detail.  The geometric rationale behind it is kind of cool.

[^1]: This is similar, but quite different from [trellis coding](https://arxiv.org/html/2406.11235v3#:~:text=tractably%20find%20the%20closest%20representable%20vector). The [goal is different](<https://arxiv.org/html/2406.11235v3#:~:text=The%20main%20focus%20of%20QTIP%C2%A0is%20on%20what%20to%20quantize%20with%20(i.e.%C2%A0TCQ)%20and%20not%20how%20to%20quantize%20(e.g.%C2%A0adaptive%20rounding%20or%20descent%20methods).>). Here, the focus is on rounding.

This is what the best rounding looks like on a face of a cube (i.e. a vector with 3 components with the max being scaled to 1), for ternary `{-1, 0, 1}`:

![Ternary rounding doesn't look like a grid. It's like Voronoi cells on a sphere.](./images/cube-face-angles-ternary.png)

And for quintary `{-2, -1, 0, 1, 2}`:

![Quintary rounding doesn't look like a grid either. It's a cool tessellation of a sphere, though.](./images/cube-face-angles-quintary.png)

(These images are best viewed from inside a cube with all faces set to use the desired image as a texture)

---

My main unsolved challenges right now are:

- Removing the need for sorting the components to find the best rounding scale
- Finding a fast enough general method to find **both** the best rounding offset *and* scale combination

# Goals

One of the goals of this is to improve the rounding algorithms used in k-quants in [`llama.cpp`](https://github.com/ggerganov/llama.cpp).

If this somehow turns out to be equivalent to what's already used in k-quants, then at least this can serve as the basis for a geometric interpretation of k-quants.

Another eventual goal is to try the effect of the "best" rounding schemes on quantization-aware training and to test if it matters or not.
