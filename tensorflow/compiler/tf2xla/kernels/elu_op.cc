/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Native XLA implementations of XLA Elu Ops

#include "tensorflow/compiler/tf2xla/kernels/elu_op.h"

#include "tensorflow/compiler/tf2xla/kernels/cwise_ops.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/types.h"

namespace xla {
XlaOp Elu(XlaOp x) {
  const auto zero = ScalarLike(x, 0);
  const auto pred = Gt(x, zero);
  const auto expm1 = Expm1(x);
  return Select(pred, x, expm1);
}

XlaOp Selu(XlaOp x) {
  const auto zero = ScalarLike(x, 0);
  const auto scale = ScalarLike(x, 1.0507009873554804934193349852946);
  const auto scale_alpha = ScalarLike(x, 1.7580993408473768599402175208123);
  const auto pred = Gt(x, zero);
  const auto expm1 = Expm1(x);
  return Select(pred, Mul(scale, x), Mul(scale_alpha, expm1));
}

XlaOp GeluApprox(XlaOp x) {
/*
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * x*(1.0 + 0.044715*x*x))))
  return x * cdf
*/
  const auto c044 = ScalarLike(x, 0.044715);
  const auto csqrt2pi = ScalarLike(x, 0.7978845608028654);
  const auto one_half = ScalarLike(x, 0.5);
  const auto one = ScalarLike(x, 1.0);
  return Mul(x, Mul(one_half, Add(one, Tanh(Mul(csqrt2pi, Mul(x, Add(one, Mul(c044,Mul(x,x)))))))));
}


XlaOp GeluGradApprox(XlaOp x) {
/*
  const T p1 = static_cast<T>(0.7978845608028654);
  const T p3 = static_cast<T>(0.044715 * 0.7978845608028654);
  T x = feature[i];
  T z = p1 * x + p3 * x * x * x;
  T g = gradient[i];
  T cz = 1. / cosh(z);
  backprop[i] = static_cast<T>(
      g * 0.5 * (1. + tanh(z) + x * (p1 + 3 * p3 * x * x) * cz * cz));

*/
  const auto one_half = ScalarLike(x, 0.5);
  auto p1 = ScalarLike(x, 0.7978845608028654);
  auto p3 = ScalarLike(x, 0.044715 * 0.7978845608028654);
  auto p3x3 = ScalarLike(x, 0.044715 * 0.7978845608028654 * 3);
  const auto z = Mul(x, Add(p1, Mul(p3, Mul(x, x))));
  const auto z3 = Mul(x, Add(p1, Mul(p3x3, Mul(x, x))));
  const auto tz = Tanh(z);
//  T cz = 1. / cosh(z);
// 1/cosh^2 z = (cosh^2 z - sinh^2 z)/cosh^2 z = 1 - tanh^2 z
  const auto one = ScalarLike(x, 1.0);
  const auto cz2 = Sub(one, Mul(tz, tz));
  return Mul(one_half, Add(one, Add(tz, Mul(z3, cz2))));
}

#if 0
XlaOp Gelu(XlaOp x) {
  const auto sqrt_half = ScalarLike(x, 1./1.4142135623730951);
  const auto one_half = ScalarLike(x, 0.5);
  const auto one = ScalarLike(x, 1.0);
  return Mul(x, Mul(one_half, Add(one, Erf(Mul(x, sqrt_half)))));
/*
      return 0.5 * features * (1.0 + math_ops.erf(
          features / math_ops.cast(1.4142135623730951, features.dtype)))
*/
}

XlaOp GeluGrad(XlaOp x) {
/*

   y = 0.5 * x * (1+erf(x*c))
   y' = 0.5 * (1+erf(x*c)) + 0.5*x*c*(2/sqrt(pi))*exp(-(x*c)^2)
  c = sqrt(0.5)
   y' = 0.5 * (1+erf(x*c)) + 0.5*x*sqrt(2/pi)*exp(-0.5*x^2)
*/
  const auto csqrt2pi = ScalarLike(x, 0.7978845608028654);
  const auto sqrt_half = ScalarLike(x, 1./1.4142135623730951);
  const auto one_half = ScalarLike(x, 0.5);
  const auto neg_one_half = ScalarLike(x, -0.5);
  const auto one = ScalarLike(x, 1.0);
  return Mul(one_half, Add(one, Add(Erf(Mul(x, sqrt_half)), Mul(x, Mul(csqrt2pi, Exp(Mul(neg_one_half,Mul(x,x))))))));
}
#endif

}  // namespace xla

namespace tensorflow {
namespace {

class EluOp : public XlaOpKernel {
 public:
  explicit EluOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  // Computes the max of the scalar input x and 0.
  void Compile(XlaOpKernelContext* ctx) override {
    ctx->SetOutput(0, xla::Elu(ctx->Input(0)));
  }
};

class EluGradOp : public XlaOpKernel {
 public:
  explicit EluGradOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  // Return the lhs (incoming gradient) if the rhs (input feature) > 0,
  // otherwise return lhs * (1 + rhs).
  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* b = ctx->builder();
    const auto zero = XlaHelpers::Zero(b, input_type(0));
    const auto one = XlaHelpers::One(b, input_type(0));
    const auto grad = ctx->Input(0);
    const auto activation = ctx->Input(1);
    const auto exp_grad = xla::Mul(grad, xla::Add(activation, one));
    const auto pred = xla::Gt(activation, zero);
    ctx->SetOutput(0, xla::Select(pred, grad, exp_grad));
  }
};

REGISTER_XLA_OP(Name("Elu"), EluOp);
REGISTER_XLA_OP(Name("EluGrad"), EluGradOp);

class SeluOp : public XlaOpKernel {
 public:
  explicit SeluOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  // Computes the max of the scalar input x and 0.
  void Compile(XlaOpKernelContext* ctx) override {
    ctx->SetOutput(0, xla::Selu(ctx->Input(0)));
  }
};

class SeluGradOp : public XlaOpKernel {
 public:
  explicit SeluGradOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  // Return the lhs (incoming gradient) if the rhs (input feature) > 0,
  // otherwise return lhs * (1 + rhs).
  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* b = ctx->builder();
    const auto zero = XlaHelpers::Zero(b, input_type(0));
    const auto scale = XlaHelpers::FloatLiteral(b, input_type(0),
            1.0507009873554804934193349852946);
    const auto scale_alpha = XlaHelpers::FloatLiteral(b, input_type(0),
            1.7580993408473768599402175208123);
    const auto grad = ctx->Input(0);
    const auto activation = ctx->Input(1);
    const auto lin_grad = xla::Mul(grad, scale);
    const auto exp_grad = xla::Mul(grad, xla::Add(activation, scale_alpha));
    const auto pred = xla::Gt(activation, zero);
    ctx->SetOutput(0, xla::Select(pred, lin_grad, exp_grad));
  }
};

REGISTER_XLA_OP(Name("Selu"), SeluOp);
REGISTER_XLA_OP(Name("SeluGrad"), SeluGradOp);

class GeluOp : public XlaOpKernel {
  bool approx_ = true;
 public:
  explicit GeluOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) 
  {
//    OP_REQUIRES_OK(context, context->GetAttr("approximate", &approx_));
  }
  void Compile(XlaOpKernelContext* ctx) override {
/*
    if(approx_)
        ctx->SetOutput(0, xla::GeluApprox(ctx->Input(0)));
    else
        ctx->SetOutput(0, xla::Gelu(ctx->Input(0)));
*/
    ctx->SetOutput(0, xla::GeluApprox(ctx->Input(0)));
  }
};

class GeluGradOp : public XlaOpKernel {
  bool approx_ = true;
 public:
  explicit GeluGradOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) 
  {
//    OP_REQUIRES_OK(context, context->GetAttr("approximate", &approx_));
  }
  void Compile(XlaOpKernelContext* ctx) override {
    const auto grad = ctx->Input(0);
    const auto activation = ctx->Input(1);
/*
    ctx->SetOutput(0, xla::Mul(grad,
      approx_ 
        ? xla::GeluGradApprox(activation)
        : xla::GeluGrad(activation)
      ));
*/
    ctx->SetOutput(0, xla::Mul(grad, xla::GeluGradApprox(activation)));
  }
};


REGISTER_XLA_OP(Name("Gelu"), GeluOp);
REGISTER_XLA_OP(Name("GeluGrad"), GeluGradOp);

}  // namespace
}  // namespace tensorflow
