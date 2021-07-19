/*
 * Copyright (C) 2021 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "MorphHelper.h"

#include <filament/BufferObject.h>
#include <filament/RenderableManager.h>
#include <filament/VertexBuffer.h>

#include "GltfEnums.h"
#include "TangentsJob.h"

using namespace filament;
using namespace filament::math;
using namespace utils;

static constexpr uint8_t kUnused = 0xff;

namespace gltfio {

uint32_t computeBindingSize(const cgltf_accessor* accessor);
uint32_t computeBindingOffset(const cgltf_accessor* accessor);

static const auto FREE_CALLBACK = [](void* mem, size_t, void*) { free(mem); };

// Returns true if a is a subset of b.
static bool isSubsetOf(ubyte4 a, ubyte4 b) {
    if (a.x != kUnused && a.x != b.x && a.x != b.y && a.x != b.z && a.x != b.w) return false;
    if (a.y != kUnused && a.y != b.x && a.y != b.y && a.y != b.z && a.y != b.w) return false;
    if (a.z != kUnused && a.z != b.x && a.z != b.y && a.z != b.z && a.z != b.w) return false;
    if (a.w != kUnused && a.w != b.x && a.w != b.y && a.w != b.z && a.w != b.w) return false;
    return true;
}

static int indexOf(int a, ubyte4 b) {
    if (a == b.x) return 0;
    if (a == b.y) return 1;
    if (a == b.z) return 2;
    if (a == b.w) return 3;
    return -1;
}

MorphHelper::MorphHelper(FFilamentAsset* asset, FFilamentInstance* inst) : mAsset(asset),
        mInstance(inst) {
    NodeMap& sourceNodes = asset->isInstanced() ? asset->mInstances[0]->nodeMap : asset->mNodeMap;
    for (auto pair : sourceNodes) {
        cgltf_node const* node = pair.first;
        cgltf_mesh const* mesh = node->mesh;
        if (mesh) {
            cgltf_primitive const* prims = mesh->primitives;
            for (cgltf_size pi = 0, count = mesh->primitives_count; pi < count; ++pi) {
                addPrimitive(mesh, pi, &mMorphTable[pair.second]);
            }
        }
    }


    auto& engine = *mAsset->mEngine;
    auto rm = &engine.getRenderableManager();
    for (auto pair: sourceNodes) {
        cgltf_node const* node = pair.first;
        cgltf_mesh const* mesh = node->mesh;
        if (mesh && mesh->weights_count > 0) {
            for (cgltf_size pi = 0, count = mesh->primitives_count; pi < count; ++pi) {
                const int W = 2048;
                const int H = 2048;
                float* pData = new float[W * H * 3];
                float* pNormal = new float[W * H * 3];

                const cgltf_primitive& prim = mesh->primitives[pi];
                for (int targetIndex = 0; targetIndex < prim.targets_count; targetIndex++) {
                    const cgltf_morph_target& morphTarget = prim.targets[targetIndex];
                    for (cgltf_size aindex = 0; aindex < morphTarget.attributes_count; aindex++) {
                        const cgltf_attribute& attribute = morphTarget.attributes[aindex];
                        const cgltf_accessor* accessor = attribute.data;
                        const cgltf_attribute_type atype = attribute.type;
                        if (atype == cgltf_attribute_type_position) {
                            if (accessor->buffer_view) {
                                auto bufferData = (const uint8_t*) accessor->buffer_view->buffer->data;
                                assert_invariant(bufferData);
                                const uint8_t* data = computeBindingOffset(accessor) + bufferData;
                                const uint32_t size = computeBindingSize(accessor);

                                int nVert = size / (3 * 4);
                                float* dataF = (float*)data;
                                float* offset;
                                for (int v = 0; v < nVert; v++) {
                                    offset = pData + prim.targets_count * v * 3;
                                    offset[3 * targetIndex + 0] = dataF[3 * v + 0];
                                    offset[3 * targetIndex + 1] = dataF[3 * v + 1];
                                    offset[3 * targetIndex + 2] = dataF[3 * v + 2];
                                }
                            }
                            continue;
                        }

                        if (atype == cgltf_attribute_type_normal) {

                            // TODO: use JobSystem for this, like what we do for non-morph tangents.
//                            TangentsJob job;
//                            TangentsJob::Params params = { .in = { &prim, targetIndex } };
//                            TangentsJob::run(&params);
//
//                            if (params.out.results) {
//                                const size_t size = params.out.vertexCount * sizeof(short4);
//
////                                BufferObject* bufferObject = BufferObject::Builder().size(size).build(engine);
////                                VertexBuffer::BufferDescriptor bd(params.out.results, size, FREE_CALLBACK);
////                                bufferObject->setBuffer(engine, std::move(bd));
////                                params.out.results = nullptr;
//                            }
                            std::unique_ptr<float3[]> morphDeltas;
                            morphDeltas.reset(new float3[accessor->count]);
                            cgltf_accessor_unpack_floats(accessor, &morphDeltas[0].x, accessor->count * 3);

                            float* offset;
                            for (int v = 0; v < accessor->count; v++) {
                                offset = pNormal + prim.targets_count * v * 3;
                                offset[3 * targetIndex + 0] = morphDeltas[v].x;
                                offset[3 * targetIndex + 1] = morphDeltas[v].y;
                                offset[3 * targetIndex + 2] = morphDeltas[v].z;
                            }

                            continue;
                        }
                    }
                }

                // Position blendshapes
                filament::Texture::PixelBufferDescriptor pBufferP(pData, size_t(W * H * 3 * sizeof(float)), Texture::Format::RGB, Texture::Type::FLOAT, [](void* buffer, size_t size, void* user) {
                    delete[] (float*)buffer;
                });
                filament::Texture *bsTextureP = filament::Texture::Builder()
                    .width(uint32_t(W))
                    .height(uint32_t(H))
                    .levels(0)
                    .sampler(filament::Texture::Sampler::SAMPLER_2D)
                    .format(filament::Texture::InternalFormat::RGB32F)
                    .build(engine);
                bsTextureP->setImage(engine, 0, std::move(pBufferP));

                // Normal blendshapes
                filament::Texture::PixelBufferDescriptor pBufferN(pNormal, size_t(W * H * 3 * sizeof(float)), Texture::Format::RGB, Texture::Type::FLOAT, [](void* buffer, size_t size, void* user) {
                    delete[] (float*)buffer;
                });
                filament::Texture *bsTextureN = filament::Texture::Builder()
                        .width(uint32_t(W))
                        .height(uint32_t(H))
                        .levels(0)
                        .sampler(filament::Texture::Sampler::SAMPLER_2D)
                        .format(filament::Texture::InternalFormat::RGB32F)
                        .build(engine);
                bsTextureN->setImage(engine, 0, std::move(pBufferN));

                MaterialInstance* mi = rm->getMaterialInstanceAt(rm->getInstance(pair.second), pi);
                mi->setParameter("blendShapeTexDims", int2{W, H});
                mi->setParameter("numBlendShape", (int)prim.targets_count);
                mi->setParameter("blendShapePTex", bsTextureP, filament::TextureSampler(TextureSampler::MinFilter::NEAREST, TextureSampler::MagFilter::NEAREST));
                mi->setParameter("blendShapeNTex", bsTextureN, filament::TextureSampler(TextureSampler::MinFilter::NEAREST, TextureSampler::MagFilter::NEAREST));

                bsTextures.push_back(bsTextureP);
                bsTextures.push_back(bsTextureN);
            }
        }


    }

}

MorphHelper::~MorphHelper() {
    auto engine = mAsset->mEngine;
    for (auto& entry : mMorphTable) {
        for (auto& prim : entry.second.primitives) {
            for (auto& target : prim.targets) {
                engine->destroy(target.bufferObject);
            }
        }
    }

    for (int i = 0; i < bsTextures.size(); i++) {
        mAsset->mEngine->destroy(bsTextures[i]);
    }
}

void MorphHelper::applyWeights(Entity entity, float const* weights, size_t count) noexcept {
    auto& engine = *mAsset->mEngine;
    auto renderableManager = &engine.getRenderableManager();
    auto renderable = renderableManager->getInstance(entity);

    // If there are 4 or fewer targets, we can simply re-use the original VertexBuffer.
    if (count <= 4) {
        float4 vec{};
        for (size_t i = 0; i < count; i++) {
            vec[i] = weights[i];
        }
        renderableManager->setMorphWeights(renderable, vec);
        return;
    }

    // We honor up to 255 weights because our set representation is a 4-tuple of bytes, with one
    // slot reserved for a sentinel value. Note that 255 is much more than the glTF min spec of 4.
    count = std::min(count, size_t(255));

    // Make a copy of the weights because we want to re-order them.
    auto& sorted = mPartiallySortedWeights;
    sorted.clear();
    sorted.insert(sorted.begin(), weights, weights + count);

    // Find the four highest weights in O(n) by doing a partial sort.
    std::nth_element(sorted.begin(), sorted.begin() + 4, sorted.end(), [](float a, float b) {
        return a > b;
    });

    // Find the "primary indices" which are the indices of the four highest weights. This is O(n).
    ubyte4 primaryIndices = {kUnused, kUnused, kUnused, kUnused};
    for (size_t index = 0, primary = 0; index < count && primary < 4; ++index) {
        const float w = weights[index];
        if (w > 0 && (w == sorted[0] || w == sorted[1] || w == sorted[2] || w == sorted[3])) {
            primaryIndices[primary++] = index;
        }
    }

    // Swap out the buffer objects for the primary indices.
    for (const auto& prim : mMorphTable[entity].primitives) {
        for (const auto& target : prim.targets) {
            const int index = indexOf(target.morphTargetIndex, primaryIndices);
            if (index > -1) {
                assert_invariant(primaryIndices[index] == target.morphTargetIndex);
                VertexBuffer* vb = prim.vertexBuffer;
                const int bufferObjectSlot = target.type == cgltf_attribute_type_position ?
                        prim.positions[index] : prim.tangents[index];

                // Slot 0 is always used for the base position so if it's getting clobbered, then
                // something is seriously wrong. Should never occur, assert just in case.
                assert_invariant(bufferObjectSlot != 0);

                vb->setBufferObjectAt(engine, bufferObjectSlot, target.bufferObject);

                // Do not break out early because there could be more than one target entry for this
                // particular target index (e.g. positions + tangents).
            }
        }
    }

    // Finally, set the 4-tuple uniform for the weight values by derefing the primary indices.
    // Note that we first create a "safe set" by replacing the unused sentinel with zero.
    float4 highest;
    for (int i = 0; i < 4; i++) {
        highest[i] = (primaryIndices[i] == kUnused) ? 0 : weights[primaryIndices[i]];
    }
    renderableManager->setMorphWeights(renderable, highest);
}

// This method copies various morphing-related data from the FilamentAsset MeshCache primitive
// (which lives in transient memory) into the MorphHelper primitive (which will stay resident).
void MorphHelper::addPrimitive(cgltf_mesh const* mesh, int primitiveIndex, TableEntry* entry) {
    auto& engine = *mAsset->mEngine;
    const cgltf_primitive& prim = mesh->primitives[primitiveIndex];
    const auto& gltfioPrim = mAsset->mMeshCache.at(mesh)[primitiveIndex];
    VertexBuffer* vertexBuffer = gltfioPrim.vertices;

    entry->primitives.push_back({ vertexBuffer });
    auto& morphHelperPrim = entry->primitives.back();

    for (int i = 0; i < 4; i++) {
        morphHelperPrim.positions[i] = gltfioPrim.morphPositions[i];
        morphHelperPrim.tangents[i] = gltfioPrim.morphTangents[i];
    }

    const cgltf_accessor* previous = nullptr;
    for (int targetIndex = 0; targetIndex < prim.targets_count; targetIndex++) {
        const cgltf_morph_target& morphTarget = prim.targets[targetIndex];
        for (cgltf_size aindex = 0; aindex < morphTarget.attributes_count; aindex++) {
            const cgltf_attribute& attribute = morphTarget.attributes[aindex];
            const cgltf_accessor* accessor = attribute.data;
            const cgltf_attribute_type atype = attribute.type;
            if (atype == cgltf_attribute_type_tangent) {
                continue;
            }
            if (atype == cgltf_attribute_type_normal) {

                // TODO: use JobSystem for this, like what we do for non-morph tangents.
                TangentsJob job;
                TangentsJob::Params params = { .in = { &prim, targetIndex } };
                TangentsJob::run(&params);

                if (params.out.results) {
                    const size_t size = params.out.vertexCount * sizeof(short4);
                    BufferObject* bufferObject = BufferObject::Builder().size(size).build(engine);
                    VertexBuffer::BufferDescriptor bd(params.out.results, size, FREE_CALLBACK);
                    bufferObject->setBuffer(engine, std::move(bd));
                    params.out.results = nullptr;
                    morphHelperPrim.targets.push_back({bufferObject, targetIndex, atype});
                }
                continue;
            }
            if (atype == cgltf_attribute_type_position) {
                // All position attributes must have the same data type.
                assert_invariant(!previous || previous->component_type == accessor->component_type);
                assert_invariant(!previous || previous->type == accessor->type);
                previous = accessor;

                // This should always be non-null, but don't crash if the glTF is malformed.
                if (accessor->buffer_view) {
                    auto bufferData = (const uint8_t*) accessor->buffer_view->buffer->data;
                    assert_invariant(bufferData);
                    const uint8_t* data = computeBindingOffset(accessor) + bufferData;
                    const uint32_t size = computeBindingSize(accessor);

                    // This creates a copy because we don't know when the user will free the cgltf
                    // source data. For non-morphed vertex buffers, we use a sharing mechanism to
                    // prevent copies, but here we just want to keep it as simple as possible.
                    uint8_t* clone = (uint8_t*) malloc(size);
                    memcpy(clone, data, size);

                    BufferObject* bufferObject = BufferObject::Builder().size(size).build(engine);
                    VertexBuffer::BufferDescriptor bd(clone, size, FREE_CALLBACK);
                    bufferObject->setBuffer(engine, std::move(bd));
                    morphHelperPrim.targets.push_back({bufferObject, targetIndex, atype});
                }
            }
        }
    }
}

}  // namespace gltfio
