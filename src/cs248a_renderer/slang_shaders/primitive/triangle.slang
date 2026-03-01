implementing "../primitive.slang";
import math;

public struct Triangle : IBoundingBox, IRayCastable
{
    public float3[3] vertices;
    public float3[3] normals;
    public float2[3] uvs;
    public float3[3] colors;
    public uint materialId;
    public uint offset;

    public BoundingBox boundingBox()
    {
        return BoundingBox(float3(0.0), float3(0.0));
    }

    /* Returns the average color of the triangle. */
    public float3 averageColor()
    {
        return (this.colors[0] + this.colors[1] + this.colors[2]) / 3.0;
    }

    /**
     * Calculates the barycentric coordinates of a point on the triangle.
     * @param p The point on the triangle to calculate the barycentric coordinates for.
     * @return The barycentric coordinates of the point.
     */
    public float3 calculateBarycentricCoord(float3 p)
    {
        const float EPSILON = 1e-6;
        float u = 0.0;
        float v = 0.0;
        float w = 0.0;


        float3 a = vertices[0];
        float3 b = vertices[1];
        float3 c = vertices[2];

        float3 ab = b - a;
        float3 ap = p - a;
        float3 ac = c - a;

        float3 abc = cross(ab, ac);

        w = length(cross(ab, ap)) / (length(abc) + EPSILON);
        v = length(cross(ac, ap)) / (length(abc) + EPSILON);
        u = 1.0 - v - w;

        return float3(u, v, w);
    }

    public float2 getUV(float3 barycentricCoord)
    {
        float2 uv = float2(0.0);

        float u = barycentricCoord.x;
        float v = barycentricCoord.y;
        float w = barycentricCoord.z;
        uv = u * this.uvs[0] + v * this.uvs[1] + w * this.uvs[2];
        
        return uv;
    }

    public float3 getColor(float3 barycentricCoord)
    {
        float u = barycentricCoord.x;
        float v = barycentricCoord.y;
        float w = barycentricCoord.z;
        float3 color = u * this.colors[0] + v * this.colors[1] + w * this.colors[2];
        return color;
    }

    public float3 getNormal(float3 barycentricCoord)
    {
        float u = barycentricCoord.x;
        float v = barycentricCoord.y;
        float w = barycentricCoord.z;
        float3 normal = u * this.normals[0] + v * this.normals[1] + w * this.normals[2];
        return normalize(normal);
    }


    public Optional<RayHitResult> hit(Ray ray)
    {
        const float EPSILON = 1e-6;
        
        
        float3 edge1 = vertices[1] - vertices[0];
        float3 edge2 = vertices[2] - vertices[0];
        
        float3 h = cross(ray.direction, edge2);
        float a = dot(edge1, h);
        
        // Check if ray is parallel to triangle
        if (abs(a) < EPSILON) {
            return Optional<RayHitResult>();
        }
        
        float f = 1.0 / a;
        float3 s = ray.origin - vertices[0];
        float u = f * dot(s, h);
        
        
        if (u < -EPSILON || u > 1.0 + EPSILON) {
            return Optional<RayHitResult>();
        }
        
        float3 q = cross(s, edge1);
        float v = f * dot(ray.direction, q);
        
        
        if (v < -EPSILON || u + v > 1.0 + EPSILON) {
            return Optional<RayHitResult>();
        }
        
        
        float t = f * dot(edge2, q);
        
        if (t < ray.tRange[0] || t > ray.tRange[1]) {
            return Optional<RayHitResult>();
        }
        
        RayHitResult result;
        result.t = t;
        result.normal = normalize(cross(edge1, edge2));
        return result;
    }

    public uint getMaterialId()
    {
        return this.materialId + this.offset;
    }
}
