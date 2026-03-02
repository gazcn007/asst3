implementing "../renderer.slang";

import light;

public struct RayMeshIntersectionResult
{
    public RayHitResult closestHit;
    public Triangle hitTriangle;
}

public Optional<RectangularLight.LightHitResult> rayRectangularLightIntersection(Ray ray, RendererUniform uniforms)
{
    Optional<RectangularLight.LightHitResult> closestLightHitResult = Optional<RectangularLight.LightHitResult>();
    for (int i = 0; i < uniforms.rectangularLightCount; i++)
    {
        Optional<RectangularLight.LightHitResult> lightHitResult = uniforms.rectangularLightBuf[i].hit(ray);
        if (lightHitResult.hasValue)
        {
            if (!closestLightHitResult.hasValue || lightHitResult.value.t < closestLightHitResult.value.t)
            {
                closestLightHitResult = lightHitResult;
            }
        }
    }
    return closestLightHitResult;
}

public Optional<RayMeshIntersectionResult> rayMeshIntersection(Ray ray, RendererUniform uniforms, bool occlusionTest = false)
{
    Optional<RayHitResult> result = Optional<RayHitResult>();
    Optional<RayMeshIntersectionResult> allResult = Optional<RayMeshIntersectionResult>();
    Optional<Triangle> hitTriangle = Optional<Triangle>();

    // IMPORTANT NOTE: Check the modified signature of `rayMeshIntersection` function in ray_mesh_intersection.slang.
    // Can you modify your existing code to make use of this flag? This is helpful in `isLightOccluded` function.

    // TODO: Student implementation starts here.
    if (uniforms.useBVH) {
        Optional<BVH<Triangle>.HitResult> bvhHitResult = uniforms.bvh.hit(ray);
        if (bvhHitResult.hasValue) {
            result = Optional<RayHitResult>(bvhHitResult.value.rayHitResult);
            hitTriangle = Optional<Triangle>(bvhHitResult.value.primitive);
        }
    } else {
        for (int i = 0; i < uniforms.triangleCount; i++){
            Triangle triangle = uniforms.triangleBuf[i];
            Optional<RayHitResult> triangleHitResult = triangle.hit(ray);
            if (triangleHitResult.hasValue){
                if (!result.hasValue || triangleHitResult.value.t < result.value.t){
                    result = triangleHitResult.value;
                    hitTriangle = Optional<Triangle>(triangle);
                }
            }
        }
    }
    

    // TODO: Student implementation ends here.

    if (result.hasValue && hitTriangle.hasValue)
    {
        allResult = Optional<RayMeshIntersectionResult>(RayMeshIntersectionResult(result.value, hitTriangle.value));
    }

    return allResult;
}
