import random
import torch
import eecs598

""" Utilities for computing and checking gradients. """


def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-7):
  """
  Utility function to perform numeric gradient checking. We use the centered
  difference formula to compute a numeric derivative:
  
  f'(x) =~ (f(x + h) - f(x - h)) / (2h)

  Rather than computing a full numeric gradient, we sparsely sample a few
  dimensions along which to compute numeric derivatives.

  Inputs:
  - f: A function that inputs a torch tensor and returns a torch scalar
  - x: A torch tensor giving the point at which to evaluate the numeric gradient
  - analytic_grad: A torch tensor giving the analytic gradient of f at x
  - num_checks: The number of dimensions along which to check
  - h: Step size for computing numeric derivatives
  """
  # fix random seed to 0 
  eecs598.reset_seed(0)
  for i in range(num_checks):
    
    ix = tuple([random.randrange(m) for m in x.shape])
    
    oldval = x[ix].item()
    x[ix] = oldval + h # increment by h
    fxph = f(x).item() # evaluate f(x + h)
    x[ix] = oldval - h # increment by h
    fxmh = f(x).item() # evaluate f(x - h)
    x[ix] = oldval     # reset

    grad_numerical = (fxph - fxmh) / (2 * h)
    grad_analytic = analytic_grad[ix]
    rel_error_top = abs(grad_numerical - grad_analytic)
    rel_error_bot = (abs(grad_numerical) + abs(grad_analytic) + 1e-12)
    rel_error = rel_error_top / rel_error_bot
    msg = 'numerical: %f analytic: %f, relative error: %e'
    print(msg % (grad_numerical, grad_analytic, rel_error))


def compute_numeric_gradient(f, x, h=1e-7):
  """ 
  Compute the numeric gradient of f at x using a finite differences
  approximation. We use the centered difference:

  df    f(x + h) - f(x - h)
  -- ~= -------------------
  dx           2 * h
  
  Inputs:
  - f: A function that inputs a torch tensor and returns a torch scalar
  - x: A torch tensor giving the point at which to compute the gradient

  Returns:
  - grad: A tensor of the same shape as x giving the gradient of f at x
  """ 
  fx = f(x) # evaluate function value at original point
  flat_x = x.contiguous().view(-1)
  grad = torch.zeros_like(x)
  flat_grad = grad.view(-1)
  # iterate over all indexes in x
  for i in range(flat_x.shape[0]):
    oldval = flat_x[i].item() # Store the original value
    flat_x[i] = oldval + h    # Increment by h
    fxph = f(x).item()        # Evaluate f(x + h)
    flat_x[i] = oldval - h    # Decrement by h
    fxmh = f(x).item()        # Evaluate f(x - h)
    flat_x[i] = oldval        # Restore original value

    # compute the partial derivative with centered formula
    flat_grad[i] = (fxph - fxmh) / (2 * h)

  return grad


def rel_error(x, y, eps=1e-10):
  """
  Compute the relative error between a pair of tensors x and y,
  which is defined as:

                          max_i |x_i - y_i]|
  rel_error(x, y) = -------------------------------
                    max_i |x_i| + max_i |y_i| + eps

  Inputs:
  - x, y: Tensors of the same shape
  - eps: Small positive constant for numeric stability

  Returns:
  - rel_error: Scalar giving the relative error between x and y
  """
  """ returns relative error between x and y """
  top = (x - y).abs().max().item()
  bot = (x.abs() + y.abs()).clamp(min=eps).max().item()
  return top / bot
