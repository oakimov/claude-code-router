"use strict";

const { createReadStream } = require("node:fs");
const probeStream = require("probe-image-size/stream");

let concurrency = 100;
let active = 0;
const waiters = [];

function drain() {
  while (active < concurrency && waiters.length > 0) {
    active += 1;
    waiters.shift()();
  }
}

function acquire() {
  if (active < concurrency) {
    active += 1;
    return Promise.resolve();
  }
  return new Promise((resolve) => waiters.push(resolve));
}

function release() {
  active -= 1;
  drain();
}

function setConcurrency(value) {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new TypeError("Concurrency must be a positive integer");
  }
  concurrency = value;
  drain();
}

async function imageSizeFromFile(filePath) {
  await acquire();
  try {
    const result = await probeStream(createReadStream(filePath));
    const dimensions = {
      width: result.width,
      height: result.height,
      type: result.type,
    };
    if (result.orientation !== undefined) {
      dimensions.orientation = result.orientation;
    }
    if (Array.isArray(result.variants)) {
      dimensions.images = result.variants.map(({ width, height }) => ({
        width,
        height,
        type: result.type,
      }));
    }
    return dimensions;
  } finally {
    release();
  }
}

module.exports = { imageSizeFromFile, setConcurrency };
