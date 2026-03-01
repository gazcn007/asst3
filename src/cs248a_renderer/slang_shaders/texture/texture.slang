implementing "../texture.slang";

public struct SharedTexture2DBuffer<T>
    where T : IFloat
{
    public StructuredBuffer<T> buffer;

    T getBufferValue(SharedTexture2D tex, uint2 texelCoord)
    {
        uint2 texSize = tex.size;
        uint offset = tex.offset;
        uint index = texelCoord.y * texSize.x + texelCoord.x;
        return buffer[offset + index];
    }

    T lerp(T a, T b, float s) { 
        return a.mul(T(1.0 - s)).add(b.mul(T(s))); 
    }

    public T pointSample(SharedTexture2D tex, float2 uv)
    {
        // TODO: Student implementation starts here.
        if (any(uv < float2(0.0)) || any(uv > float2(1.0)))
            return T(0.0);
        uint2 texelCoord = uint2(uv * tex.size);
        return getBufferValue(tex, texelCoord);
        // TODO: Student implementation ends here.
    }

    public T bilinearSample(SharedTexture2D tex, float2 uv)
    {
        // TODO: Student implementation starts here.
        if (any(uv < float2(0.0)) || any(uv > float2(1.0)))
            return T(0.0);
        
        float2 texelCenter = uv * float2(tex.size) - float2(0.5);
        uint2 baseTexelCoord = uint2(max(float2(0), floor(texelCenter)));
        float2 t = texelCenter - float2(baseTexelCoord);
        
        uint2 maxCoord = tex.size - uint2(1);
        uint2 coord00 = min(baseTexelCoord, maxCoord);
        uint2 coord10 = min(baseTexelCoord + uint2(1, 0), maxCoord);
        uint2 coord01 = min(baseTexelCoord + uint2(0, 1), maxCoord);
        uint2 coord11 = min(baseTexelCoord + uint2(1, 1), maxCoord);
        
        return lerp(
            lerp(getBufferValue(tex, coord00), getBufferValue(tex, coord10), t.x),
            lerp(getBufferValue(tex, coord01), getBufferValue(tex, coord11), t.x),
            t.y
        );
        // TODO: Student implementation ends here.
    }

    public T trilinearSample(SharedTexture2D[] mipmapLevels, float2 uv, float d, uint totalLevels)
    {
        // TODO: Student implementation starts here.
        if (any(uv < float2(0.0)) || any(uv > float2(1.0)))
            return T(0.0);
        // zero padding d based on 0 and totalLevels - 1
        float currentLevel = d;
        float previousLevel = max(d - 1, 0);
        float nextLevel = min(d + 1, totalLevels - 1);
        T currentLevelBilinear = bilinearSample(mipmapLevels[uint(currentLevel)], uv);
        T previousLevelBilinear = bilinearSample(mipmapLevels[uint(previousLevel)], uv);
        T nextLevelBilinear = bilinearSample(mipmapLevels[uint(nextLevel)], uv);
        return lerp(
            lerp(currentLevelBilinear, previousLevelBilinear, (currentLevel - previousLevel) / (nextLevel - previousLevel)),
            lerp(currentLevelBilinear, nextLevelBilinear, (currentLevel - nextLevel) / (previousLevel - nextLevel)),
            (currentLevel - previousLevel) / (nextLevel - previousLevel)
        );
        // TODO: Student implementation ends here.
    }
}

public struct SharedTexture2D
{
    public uint2 size;
    public uint offset;
}

public struct SharedTexture3DBuffer<T>
    where T : IFloat
{
    public StructuredBuffer<T> buffer;

    T getBufferValue(SharedTexture3D tex, uint3 voxelCoord)
    {
        uint3 texSize = tex.size;
        uint offset = tex.offset;
        uint index = voxelCoord.z * texSize.y * texSize.x +
                     voxelCoord.y * texSize.x +
                     voxelCoord.x;
        return buffer[offset + index];
    }

    T lerp(T a, T b, float s) { 
        return a.mul(T(1.0 - s)).add(b.mul(T(s))); 
    }

    /**
     * Sample the texture at the given uvw coordinates using nearest neighbor interpolation.
     * @param tex The texture to sample.
     * @param uvw The uvw coordinates to sample the texture at.
     * @return The sampled value.
     */
    public T pointSample(SharedTexture3D tex, float3 uvw)
    {
        // TODO: Student implementation starts here.
        if (any(uvw < float3(0.0)) || any(uvw > float3(1.0)))
            return T(0.0);
        uint3 voxelCoord = uint3(uvw * tex.size);
        return getBufferValue(tex, voxelCoord);


        // TODO: Student implementation ends here.
    }

    /**
     * Sample the texture at the given uvw coordinates using trilinear interpolation.
     * @param tex The texture to sample.
     * @param uvw The uvw coordinates to sample the texture at.
     * @return The sampled value.
     **/
    public T trilinearSample(SharedTexture3D tex, float3 uvw)
    {
        // TODO: Student implementation starts here.
        if (any(uvw < float3(0.0)) || any(uvw > float3(1.0)))
            return T(0.0);
        float3 voxelCenter = uvw * float3(tex.size);
        uint3 baseVoxelCoord = uint3(max(float3(0), floor(voxelCenter)));
        float3 t = voxelCenter - float3(baseVoxelCoord);
        return lerp(
            lerp(
                lerp(
                    getBufferValue(tex, baseVoxelCoord + uint3(0, 0, 0)),
                    getBufferValue(tex, baseVoxelCoord + uint3(1, 0, 0)),
                    t.x
                ),
                lerp(
                    getBufferValue(tex, baseVoxelCoord + uint3(0, 1, 0)),
                    getBufferValue(tex, baseVoxelCoord + uint3(1, 1, 0)),
                    t.x
                ),
                t.y
            ),
            lerp(
                lerp(
                    getBufferValue(tex, baseVoxelCoord + uint3(0, 0, 1)),
                    getBufferValue(tex, baseVoxelCoord + uint3(1, 0, 1)),
                    t.x
                ),
                lerp(
                    getBufferValue(tex, baseVoxelCoord + uint3(0, 1, 1)),
                    getBufferValue(tex, baseVoxelCoord + uint3(1, 1, 1)),
                    t.x
                ),
                t.y
            ),
            t.z
        );
        // TODO: Student implementation ends here.
    }
}

public struct SharedTexture3D
{
    public uint3 size;
    public uint offset;
}

