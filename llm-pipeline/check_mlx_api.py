from mlx_lm import generate
import inspect

print("mlx_lm.generate() signature:")
print(inspect.signature(generate))
print("\nDocstring:")
print(generate.__doc__)
