// Copyright © 2023 Apple Inc.

#include <numeric>

#include "doctest/doctest.h"

#include "mlx/mlx.h"

using namespace mlx::core;

TEST_CASE("test matmul") {
  auto a = array(1);
  auto b = array({1.0});
  CHECK_THROWS_AS(matmul(a, b), std::invalid_argument);

  a = array({1.0});
  b = array({1.0});
  auto out = matmul(a, b);
  CHECK_EQ(out.shape(), Shape{});
  CHECK_EQ(out.size(), 1);
  CHECK_EQ(out.dtype(), float32);
  CHECK_EQ(out.item<float>(), 1.0f);

  a = ones({2, 4});
  b = ones({2});
  CHECK_THROWS_AS(matmul(a, b), std::invalid_argument);

  a = ones({2, 4});
  b = ones({3, 2});
  CHECK_THROWS_AS(matmul(a, b), std::invalid_argument);

  a = ones({2, 4});
  b = ones({4, 3, 2});
  CHECK_THROWS_AS(matmul(a, b), std::invalid_argument);

  a = ones({2});
  b = ones({4, 2});
  CHECK_THROWS_AS(matmul(a, b), std::invalid_argument);

  a = ones({2, 3});
  b = ones({4, 2});
  CHECK_THROWS_AS(matmul(a, b), std::invalid_argument);

  a = ones({2, 4, 3});
  b = ones({4, 2});
  CHECK_THROWS_AS(matmul(a, b), std::invalid_argument);

  a = ones({2, 4});
  b = ones({4, 2});
  out = matmul(a, b);
  CHECK(array_equal(out, full({2, 2}, 4.0f)).item<bool>());

  a = ones({2, 4}, int32);
  b = ones({4, 2}, float32);
  out = matmul(a, b);
  CHECK(array_equal(out, full({2, 2}, 4.0f)).item<bool>());

  // Check single dimensions
  a = ones({4});
  b = ones({4, 2});
  out = matmul(a, b);
  CHECK(array_equal(out, full({2}, 4.0f)).item<bool>());

  a = ones({2, 4});
  b = ones({4});
  out = matmul(a, b);
  CHECK(array_equal(out, full({2}, 4.0f)).item<bool>());

  a = ones({4});
  b = ones({4});
  out = matmul(a, b);
  CHECK(array_equal(out, full({}, 4.0f)).item<bool>());

  // Test transposed arrays
  a = array({1.0f, 1.0f, 1.0f, 1.0f}, {1, 4});
  b = array({1.0f, 1.0f, 1.0f, 1.0f}, {4, 1});
  out = matmul(transpose(a), transpose(b));
  CHECK(array_equal(out, ones({4, 4})).item<bool>());

  a = array({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  b = array({1.0f, 2.0f, 1.0f, 2.0f}, {2, 2});
  out = matmul(transpose(a), b);
  CHECK(
      array_equal(out, array({4.0f, 8.0f, 6.0f, 12.0f}, {2, 2})).item<bool>());

  out = matmul(a, transpose(b));
  CHECK(
      array_equal(out, array({5.0f, 5.0f, 11.0f, 11.0f}, {2, 2})).item<bool>());

  out = matmul(transpose(a), transpose(b));
  CHECK(
      array_equal(out, array({7.0f, 7.0f, 10.0f, 10.0f}, {2, 2})).item<bool>());

  // Test broadcasting for both arrays
  a = ones({5, 4, 2});
  b = ones({2, 3});
  out = matmul(a, b);
  CHECK(array_equal(out, full({5, 4, 3}, 2.0f)).item<bool>());

  a = ones({5, 1, 4, 2});
  b = ones({1, 7, 2, 3});
  out = matmul(a, b);
  CHECK(array_equal(out, full({5, 7, 4, 3}, 2.0f)).item<bool>());

  // Test batched matmul with transpose
  a = ones({2, 2, 4});
  b = ones({2, 4, 2});
  out = matmul(transpose(a, {0, 2, 1}), transpose(b, {0, 2, 1}));
  CHECK(array_equal(out, full({2, 4, 4}, 2.0f)).item<bool>());
}

TEST_CASE("test addmm") {
  // Basic addmm test: out = alpha * (A @ B) + beta * C
  // A is 2x3, B is 3x2, C is 2x2, out is 2x2
  auto a = array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
  auto b = array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2});
  auto c = ones({2, 2});

  // Test with alpha=1, beta=0 (should be same as matmul)
  auto out = addmm(c, a, b, 1.0f, 0.0f);
  auto expected = matmul(a, b);
  CHECK(allclose(out, expected).item<bool>());

  // Test with alpha=1, beta=1 (matmul + c)
  out = addmm(c, a, b, 1.0f, 1.0f);
  expected = add(matmul(a, b), c);
  CHECK(allclose(out, expected).item<bool>());

  // Test with alpha=0.5, beta=2.0
  out = addmm(c, a, b, 0.5f, 2.0f);
  expected = add(multiply(matmul(a, b), array(0.5f)), multiply(c, array(2.0f)));
  CHECK(allclose(out, expected).item<bool>());

  // Test with ones matrices
  a = ones({2, 4});
  b = ones({4, 3});
  c = full({2, 3}, 10.0f);
  out = addmm(c, a, b, 1.0f, 1.0f);
  // A @ B = 4 * ones(2,3), then + 10 = 14
  expected = full({2, 3}, 14.0f);
  CHECK(allclose(out, expected).item<bool>());

  // Test batched addmm
  a = ones({2, 3, 4});
  b = ones({2, 4, 5});
  c = full({2, 3, 5}, 2.0f);
  out = addmm(c, a, b, 0.5f, 1.0f);
  // A @ B = 4 * ones(2,3,5), then * 0.5 = 2, then + 2 = 4
  expected = full({2, 3, 5}, 4.0f);
  CHECK(allclose(out, expected).item<bool>());

  // Test empty matmul (K=0) - should return beta * C
  a = ones({2, 0});
  b = ones({0, 2});
  c = ones({2, 2});
  out = addmm(c, a, b, 0.5f, 2.0f);
  // A @ B = 0 for empty matrices, so result = beta * C = 2 * ones(2,2)
  expected = full({2, 2}, 2.0f);
  CHECK(allclose(out, expected).item<bool>());

  // Test different alpha/beta combinations with K=0
  out = addmm(c, a, b, 1.0f, 3.0f);
  expected = full({2, 2}, 3.0f);
  CHECK(allclose(out, expected).item<bool>());

  // Test broadcasting of c
  a = ones({3, 4});
  b = ones({4, 5});
  c = full({1}, 10.0f);  // scalar c broadcast to output shape
  out = addmm(c, a, b, 1.0f, 1.0f);
  // A @ B = 4 * ones(3,5), then + 10 = 14
  expected = full({3, 5}, 14.0f);
  CHECK(allclose(out, expected).item<bool>());
}
