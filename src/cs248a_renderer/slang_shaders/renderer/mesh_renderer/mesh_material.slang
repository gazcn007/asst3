implementing "../renderer.slang";

import utils;

public float2 getShiftedTexUV(float2 uv, RendererUniform uniforms, Triangle triangle, float2 delta, float2 tex_uv)
{
    float2 shifted_uv = uv + delta;
    Ray ray = uniforms.camera.generateRay(shifted_uv);
    Optional<RayHitResult> hitResult = triangle.hit(ray);
    float2 result = tex_uv;
    if (hitResult.hasValue)
    {
        float3 hitPoint = ray.origin + hitResult.value.t * ray.direction;
        float3 barycentricCoord = triangle.calculateBarycentricCoord(hitPoint);
        return triangle.getUV(barycentricCoord);
    }

    Optional<RayMeshIntersectionResult> rayMeshIntersectionResult_dx = rayMeshIntersection(ray, uniforms);
    if (rayMeshIntersectionResult_dx.hasValue)
    {
        Triangle hitTriangle = rayMeshIntersectionResult_dx.value.hitTriangle;
        if (hitTriangle.getMaterialId() == triangle.getMaterialId())
        {
            float3 hitPoint = ray.origin + rayMeshIntersectionResult_dx.value.closestHit.t * ray.direction;
            float3 barycentricCoord = hitTriangle.calculateBarycentricCoord(hitPoint);
            return hitTriangle.getUV(barycentricCoord);
        }
    }

    return tex_uv;
}

// Compute the level of the texture at the hit point.
// @param uv: The uv coordinates of the hit point.
// @param uniforms: The renderer uniforms.
// @param triangle: The triangle that was hit.
// @param tex_uv: The uv coordinates of the texture at the hit point.
// @return: The level of the texture at the hit point.
public float getLevel(float2 uv, RendererUniform uniforms, Triangle triangle, float2 tex_uv)
{
    float level = 0.0;

    // TODO: Student implementation starts here.
    
    // Compute pixel size in normalized screen coordinates
    float dx = 1.0 / float(uniforms.camera.canvasSize.x);
    float dy = 1.0 / float(uniforms.camera.canvasSize.y);
    
    // Ray differential method: shoot 4 rays (∂x⁺, ∂x⁻, ∂y⁺, ∂y⁻)
    // Use central differences for more accurate derivative estimation
    
    // shoot rays offset in +x and -x direction to compute ρₓ
    float2 texUV_px_pos = getShiftedTexUV(uv, uniforms, triangle, float2(dx, 0.0), tex_uv);
    float2 texUV_px_neg = getShiftedTexUV(uv, uniforms, triangle, float2(-dx, 0.0), tex_uv);
    float2 duvdx = (texUV_px_pos - texUV_px_neg) / (2.0 * dx);
    float rho_x = length(duvdx);
    
    // shoot rays offset in +y and -y direction to compute ρᵧ
    float2 texUV_py_pos = getShiftedTexUV(uv, uniforms, triangle, float2(0.0, dy), tex_uv);
    float2 texUV_py_neg = getShiftedTexUV(uv, uniforms, triangle, float2(0.0, -dy), tex_uv);
    float2 duvdy = (texUV_py_pos - texUV_py_neg) / (2.0 * dy);
    float rho_y = length(duvdy);
    
    // D = log₂(max(ρₓ, ρᵧ))
    float rho = max(rho_x, rho_y);
    level = log2(max(rho, 1e-6));  // Avoid log2(0)
    
    // TODO: Student implementation ends here.
    return level;
}   


public PhysicsBasedParameters getHitPointParameters(float2 uv, RendererUniform uniforms, Triangle triangle, float3 normal, float3 barycentricCoord, float3 hitPoint)
{
    float2 tex_uv = triangle.getUV(barycentricCoord);
    float level = getLevel(uv, uniforms, triangle, tex_uv);
    uint materialId = triangle.materialId + triangle.offset;
    PhysicsBasedMaterial physicsBasedMaterial = uniforms.physicsBasedMaterialBuf[materialId];
    PhysicsBasedParameters parameters = sampleMaterial(physicsBasedMaterial, uniforms.physicsBasedMaterialTextureBuf, tex_uv, level, uniforms.visualizeLevelOfDetail);
    if (uniforms.visualizeBarycentricCoords)
    {
        parameters.albedo = barycentricCoord;
    }
    if (uniforms.visualizeTexUV)
    {
        parameters.albedo = float3(tex_uv, 0.0);
    }
    return parameters;
}
