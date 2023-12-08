// Copyright (c) Megvii Inc.
// Licensed under Apache License, Version 2.0
// ===========================================================================
// Modification Copyright 2022 ByteDance Ltd. and/or its affiliates.

#include <gtest/gtest.h>
#include <thread>
#include <vector>

#include "brt/core/common/status.h"
#include "brt/core/distributed/rendezvous_socket.h"

using namespace brt::common;

TEST(TestRendezvousSocket, GetFreePort) {
  int port = brt::GetFreePort();
  ASSERT_TRUE(port > 0);
}

TEST(TestRendezvousSocket, Connect) {
  const int nranks = 3;

  const int port = brt::GetFreePort();
  auto ret = brt::CreateServer(nranks, port);
  ASSERT_EQ(Status::OK(), ret);

  auto run = [nranks, port](int rank) {
    auto client = std::make_unique<brt::RendezvousSocket>(nranks, rank);
    auto ret = client->connect("localhost", port);
    ASSERT_EQ(Status::OK(), ret);
  };

  std::vector<std::thread> threads;
  for (size_t i = 0; i < nranks; i++) {
    threads.push_back(std::thread(run, i));
  }

  for (size_t i = 0; i < nranks; i++) {
    threads[i].join();
  }
}

TEST(TestRendezvousSocket, Barrier) {
  const int nranks = 3;

  const int port = brt::GetFreePort();
  auto ret = brt::CreateServer(nranks, port);
  ASSERT_EQ(Status::OK(), ret);

  int counter = 0;

  auto run = [nranks, port, &counter](int rank) {
    auto client = std::make_unique<brt::RendezvousSocket>(nranks, rank);
    auto ret = client->connect("localhost", port);
    ASSERT_EQ(Status::OK(), ret);

    ret = client->barrier();
    ASSERT_EQ(Status::OK(), ret);

    sleep(rank);
    ++counter;

    ret = client->barrier();
    ASSERT_EQ(Status::OK(), ret);

    // if the barrier is not working correctly, threads that sleep
    // less seconds will arrive here earlier and counter might be
    // less than nranks
    ASSERT_EQ(nranks, counter);
  };

  std::vector<std::thread> threads;
  for (size_t i = 0; i < nranks; i++) {
    threads.push_back(std::thread(run, i));
  }

  for (size_t i = 0; i < nranks; i++) {
    threads[i].join();
  }
}

TEST(TestRendezvousSocket, Broadcast) {
  const int nranks = 3;
  const int root = 1;
  const int chunk_size = 10;

  const int port = brt::GetFreePort();
  auto ret = brt::CreateServer(nranks, port);
  ASSERT_EQ(Status::OK(), ret);

  std::string str(chunk_size * nranks, '\0');
  for (size_t i = 0; i < str.size(); i++) {
    str[i] = 'a' + i % 26;
  }
  auto expected = str.substr(root * chunk_size, chunk_size);

  auto run = [nranks, port, &str, &expected](int rank) {
    auto client = std::make_unique<brt::RendezvousSocket>(nranks, rank);
    auto ret = client->connect("localhost", port);
    ASSERT_EQ(Status::OK(), ret);

    const char *input = str.data() + rank * chunk_size;
    char *output = (char *)malloc(chunk_size);
    ret = client->broadcast(input, output, chunk_size, root);
    ASSERT_EQ(Status::OK(), ret);

    ASSERT_EQ(expected, std::string(output, chunk_size));
    free(output);
  };

  std::vector<std::thread> threads;
  for (size_t i = 0; i < nranks; i++) {
    threads.push_back(std::thread(run, i));
  }

  for (size_t i = 0; i < nranks; i++) {
    threads[i].join();
  }
}

TEST(TestServerClient, AllGather) {
  const int nranks = 3;
  const int chunk_size = 10;

  const int port = brt::GetFreePort();
  auto ret = brt::CreateServer(nranks, port);
  ASSERT_EQ(Status::OK(), ret);

  std::string str(chunk_size * nranks, '\0');
  for (size_t i = 0; i < str.size(); i++) {
    str[i] = 'a' + i % 26;
  }

  auto run = [nranks, port, &str](int rank) {
    auto client = std::make_unique<brt::RendezvousSocket>(nranks, rank);
    auto ret = client->connect("localhost", port);
    ASSERT_EQ(Status::OK(), ret);

    const char *input = str.data() + rank * chunk_size;
    char *output = (char *)malloc(str.size());
    ret = client->allgather(input, output, chunk_size);
    ASSERT_EQ(Status::OK(), ret);

    ASSERT_EQ(str, std::string(output, str.size()));
    free(output);
  };

  std::vector<std::thread> threads;
  for (size_t i = 0; i < nranks; i++) {
    threads.push_back(std::thread(run, i));
  }

  for (size_t i = 0; i < nranks; i++) {
    threads[i].join();
  }
}

TEST(TestServerClient, Sequence) {
  const int nranks = 3;
  const int chunk_size = 10;

  const int port = brt::GetFreePort();
  auto ret = brt::CreateServer(nranks, port);
  ASSERT_EQ(Status::OK(), ret);

  std::string str(chunk_size * nranks, '\0');
  for (size_t i = 0; i < str.size(); i++) {
    str[i] = 'a' + i % 26;
  }

  auto run = [nranks, port, &str](int rank) {
    auto client = std::make_unique<brt::RendezvousSocket>(nranks, rank);
    auto ret = client->connect("localhost", port);
    ASSERT_EQ(Status::OK(), ret);

    const char *input = str.data() + rank * chunk_size;
    char *output = (char *)malloc(str.size());

    // send a sequence of requets without checking output
    ASSERT_EQ(Status::OK(), client->barrier());
    ASSERT_EQ(Status::OK(), client->broadcast(input, output, chunk_size, 1));
    ASSERT_EQ(Status::OK(), client->allgather(input, output, chunk_size));
    ASSERT_EQ(Status::OK(), client->barrier());
    ASSERT_EQ(Status::OK(), client->allgather(input, output, chunk_size));
    ASSERT_EQ(Status::OK(), client->broadcast(input, output, chunk_size, 2));
    ASSERT_EQ(Status::OK(), client->allgather(input, output, chunk_size));
    ASSERT_EQ(Status::OK(), client->barrier());

    free(output);
  };

  std::vector<std::thread> threads;
  for (size_t i = 0; i < nranks; i++) {
    threads.push_back(std::thread(run, i));
  }

  for (size_t i = 0; i < nranks; i++) {
    threads[i].join();
  }
}
