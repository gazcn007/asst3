implementing "../renderer.slang";

import math;
import texture;

public static uint MAX_STEPS = 1024;
public static float EPSILON = 1e-6;

public static float STEP_SIZE = 0.01;
public static float DENSITY_SCALE = 5.0;

/**
 * Perform ray marching through the volume and accumulate color.
 * @param uv The normalized uv coordinate from (0, 0) in the bottom-left to (1, 1) in the top-right.
 * @param uniforms The renderer uniform containing camera, sampling, and scene information.
 * @param tMax The maximum t value for ray marching (usually the nearest primitive intersection t).
 * @param primColor The color of the nearest primitive intersection.
 * @return The accumulated color after ray marching through the volume.
 */
public float4 volumeSample(float2 uv, RendererUniform uniforms, float tMax, float4 primColor)
{
    // TODO: Student implementation starts here.
    Ray ray = uniforms.camera.generateRay(uv);
    Ray localRay = ray.transform(uniforms.volume.invModelMatrix);
    BoundingBox volumeBound = uniforms.volume.bound;
    DiffTexture3D<float, 4> data = uniforms.volume.tex;
    float4 backgroundColor = primColor;
    float4 volumeColor = accumulateColor(localRay, volumeBound, MAX_STEPS, STEP_SIZE, DENSITY_SCALE, tMax, backgroundColor, data);
    return volumeColor;
    
    // TODO: Student implementation ends here.
}

/**
 * Backward pass for volumetric ray marching at a single uv.
 * @param uv The normalized uv coordinate from (0, 0) in the bottom-left to (1, 1) in the top-right.
 * @param uniforms The renderer uniform containing camera, sampling, and scene information.
 * @param outGrad The output gradient w.r.t. the sampled color.
 */
public void volumeSampleBwd(float2 uv, RendererUniform uniforms, float4 outGrad)
{
    // TODO: Student implementation starts here.
    Ray ray = uniforms.camera.generateRay(uv);
    Ray localRay = ray.transform(uniforms.volume.invModelMatrix);
    
    float4 backgroundColor = uniforms.ambientColor;

    var diffTexPair = DifferentialPtrPair<DiffTexture3D<float, 4>>(uniforms.volume.tex, uniforms.volume.dTex);
    bwd_diff(accumulateColor)(localRay, uniforms.volume.bound, MAX_STEPS, STEP_SIZE, DENSITY_SCALE, float.maxValue, backgroundColor, diffTexPair, outGrad);
    
    // TODO: Student implementation ends here.
}

/**
 * Ray march through the volume and accumulate color.
 * @param ray The ray in local space.
 * @param volumeBound The bounding box of the volume in local space.
 * @param maxSteps The maximum number of marching steps.
 * @param stepSize The size of each marching step.
 * @param densityScale The scale factor for the volume density (alpha channel).
 * @param backgroundColor The background color behind the volume
                          (this can be the ambient color if there's no primitive behind the volume, or the color of the primitive hit if there's one).
 * @param data The 3D texture storing the volume data (rgba).
 * @return The accumulated color after ray marching through the volume.
 */
[Differentiable]
public float4 accumulateColor(
    no_diff Ray ray,
    no_diff BoundingBox volumeBound,
    no_diff uint maxSteps,
    no_diff float stepSize,
    no_diff float densityScale,
    no_diff float tMax,
    no_diff float4 backgroundColor,
    DiffTexture3D<float, 4> data)
{
    // TODO: Student implementation starts here.
    float2 hitRange = volumeBound.hit(ray);

    // bound check
    if (hitRange.y < hitRange.x)
    {
        return backgroundColor;
    }

    float3 accumulatedColor = float3(0.0, 0.0, 0.0);
    float opacity = 0.0;

    float t = max(hitRange.x, ray.tRange.x);
    float tEnd = min(hitRange.y, min(ray.tRange.y, tMax));

    [MaxIters(1024)]
    for (uint step = 0; step < maxSteps && t <= tEnd; step++) {
        float3 pos = ray.origin + t * ray.direction;

        float3 uvw = (pos - volumeBound.pMin) / (volumeBound.pMax - volumeBound.pMin);

        float4 sampleColor = sampleTrilinear(data, uvw);
        sampleColor = clamp(sampleColor, 0.0, 1.0);

        accumulatedColor += sampleColor.rgb * (1.0 - opacity) * sampleColor.a * densityScale * stepSize;
        opacity += sampleColor.a * densityScale * stepSize * (1.0 - opacity);

        t += stepSize;

        if(opacity >= 1.0 - EPSILON)
        {
            break;
        }
    }

    float3 finalColor = accumulatedColor + (1.0 - opacity) * backgroundColor.rgb;
    return float4(finalColor, opacity);
    // TODO: Student implementation ends here.
}