# Comparing `Corrosiff` and `siffreadermodule`

Was reimplementing everything in `Rust` really worth it?
Does the extra hassle of `PyO3` add much overhead? Let's find out!
Here we benchmark:

1) The performance of `CorrosiffPy` vs `siffreadermodule` calling
the same functions on the same datasets

2) The performance of `CorrosiffPy` vs the raw `Corrosiff` library

Or... we will. When I finally set these up.