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

#include <jni.h>

#include <gltfio/MaterialProvider.h>

#include <utils/debug.h>

#include "MaterialKey.h"

using namespace filament;
using namespace gltfio;

extern "C" JNIEXPORT jlong JNICALL
Java_com_google_android_filament_gltfio_UbershaderLoader_nCreateUbershaderLoader(JNIEnv*, jclass,
        jlong nativeEngine) {
    Engine* engine = (Engine*) nativeEngine;
    return (jlong) createUbershaderLoader(engine);
}

extern "C" JNIEXPORT void JNICALL
Java_com_google_android_filament_gltfio_UbershaderLoader_nDestroyUbershaderLoader(JNIEnv*, jclass,
        jlong nativeProvider) {
    auto provider = (MaterialProvider*) nativeProvider;
    delete provider;
}

extern "C" JNIEXPORT void JNICALL
Java_com_google_android_filament_gltfio_UbershaderLoader_nDestroyMaterials(JNIEnv*, jclass,
        jlong nativeProvider) {
    auto provider = (MaterialProvider*) nativeProvider;
    provider->destroyMaterials();
}

extern "C" JNIEXPORT long JNICALL
Java_com_google_android_filament_gltfio_UbershaderLoader_nCreateMaterialInstance(JNIEnv* env, jclass,
        jlong nativeProvider, jobject materialKey, jintArray uvmap, jstring label) {
    MaterialKey nativeKey = {};

    auto& helper = MaterialKeyHelper::get();
    helper.copy(env, nativeKey, materialKey);

    const char* nativeLabel = label ? env->GetStringUTFChars(label, nullptr) : nullptr;
    UvMap nativeUvMap = {};
    auto provider = (MaterialProvider*) nativeProvider;
    MaterialInstance* instance = provider->createMaterialInstance(&nativeKey, &nativeUvMap, nativeLabel);

    // Copy the UvMap results from the native array into the JVM array.
    jint* elements = env->GetIntArrayElements(uvmap, nullptr);
    if (elements) {
        const size_t javaSize = env->GetArrayLength(uvmap);
        for (int i = 0, n = std::min(javaSize, nativeUvMap.size()); i < n; ++i) {
            elements[i] = nativeUvMap[i];
        }
        env->ReleaseIntArrayElements(uvmap, elements, JNI_ABORT);
    }

    // The config parameter is an in-out parameter so we need to copy the results back to Java.
    helper.copy(env, materialKey, nativeKey);

    if (label) {
        env->ReleaseStringUTFChars(label, nativeLabel);
    }

    return (long) instance;
}

extern "C" JNIEXPORT int JNICALL
Java_com_google_android_filament_gltfio_UbershaderLoader_nGetMaterialCount(JNIEnv*, jclass,
        jlong nativeProvider) {
    auto provider = (MaterialProvider*) nativeProvider;
    return provider->getMaterialsCount();
}

extern "C" JNIEXPORT void JNICALL
Java_com_google_android_filament_gltfio_UbershaderLoader_nGetMaterials(JNIEnv* env, jclass,
        jlong nativeProvider, jlongArray result) {
    auto provider = (MaterialProvider *) nativeProvider;
    auto materials = provider->getMaterials();
    jlong *resultElements = env->GetLongArrayElements(result, nullptr);
    if (resultElements) {
        const size_t javaSize = env->GetArrayLength(result);
        for (int i = 0, n = std::min(javaSize, provider->getMaterialsCount()); i < n; ++i) {
            resultElements[i] = (jlong) materials[i];
        }
        env->ReleaseLongArrayElements(result, resultElements, JNI_ABORT);
    }
}
