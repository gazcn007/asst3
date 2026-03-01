implementing "../primitive.slang";
import math;
import texture;

// The size of each marching step.
static float STEP_SIZE = 0.005;
// The density threshold to consider a hit.
static float DENSITY_THRESHOLD = 0.25;
// The delta used for finite difference normal estimation.
static float FINITE_DIFF_DELTA = 0.001;

public struct Volume
{
    public BoundingBox bound;
    public SharedTexture3D tex;
    public float4x4 modelMatrix;
    public float4x4 invModelMatrix;

    /**
     * Test ray-volume bounding box intersection.
     * @param ray The ray to test against the volume.
     * @param volumeTexBuf The texture buffer contains all the volumetrically represented models in the scene .
     * @return An Optional containing RayHitResult if the ray hits the volume, otherwise an empty Optional.
     */
    float3 computeNormal(float3 uvw, SharedTexture3DBuffer<float4> volumeTexBuf) {
        float g_x = (volumeTexBuf.trilinearSample(tex, float3(uvw.x + FINITE_DIFF_DELTA, uvw.y, uvw.z)) 
                    - volumeTexBuf.trilinearSample(tex, float3(uvw.x - FINITE_DIFF_DELTA, uvw.y, uvw.z))).w;
        float g_y = (volumeTexBuf.trilinearSample(tex, float3(uvw.x, uvw.y + FINITE_DIFF_DELTA, uvw.z)) 
                    - volumeTexBuf.trilinearSample(tex, float3(uvw.x, uvw.y - FINITE_DIFF_DELTA, uvw.z))).w;
        float g_z = (volumeTexBuf.trilinearSample(tex, float3(uvw.x, uvw.y, uvw.z + FINITE_DIFF_DELTA)) 
                    - volumeTexBuf.trilinearSample(tex, float3(uvw.x, uvw.y, uvw.z  - FINITE_DIFF_DELTA))).w;
        return -normalize(float3(g_x, g_y, g_z));
    }

    public Optional<RayHitResult> hit(Ray ray, SharedTexture3DBuffer<float4> volumeTexBuf)
    {
        // TODO: Student implementation starts here.
        Ray localRay = ray.transform(invModelMatrix);
        for (float t = localRay.tRange.x; t < localRay.tRange.y; t += STEP_SIZE) {
            float3 p = localRay.origin + t * localRay.direction;
            float3 uvw = getUVW(p);
            float4 sample = volumeTexBuf.trilinearSample(tex, uvw);
            if (sample[3] > DENSITY_THRESHOLD) {
                float3 normal = computeNormal(uvw, volumeTexBuf);
                return Optional<RayHitResult>(RayHitResult(t, normal));
            }
        }
        // TODO: Student implementation ends here.

        return Optional<RayHitResult>();
    }

    /**
     * Get the uvw coordinates of a point in local space.
     * @param p The point in local space.
     * @return The uvw coordinates in [0, 1].
     */
    public float3 getUVW(float3 p)
    {
        // Compute uvw coordinates in [0, 1].
        float3 minBound = this.bound.pMin;
        float3 maxBound = this.bound.pMax;
        return (p - minBound) / (maxBound - minBound);
    }
};
