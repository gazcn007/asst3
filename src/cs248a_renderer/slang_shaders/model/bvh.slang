implementing "../model.slang";
import math;

public struct BVHNode
{
    public BoundingBox bound;
    public int left;
    public int right;
    public int primLeft;
    public int primRight;
    public int depth;

    public __init(BoundingBox bound, int left = -1, int right = -1, int primLeft = -1, int primRight = -1, int depth = -1)
    {
        this.bound = bound;
        this.left = left;
        this.right = right;
        this.primLeft = primLeft;
        this.primRight = primRight;
        this.depth = depth;
    }

    public float2 hit(Ray ray)
    {
        return this.bound.hit(ray);
    }
}

public struct BVH<T>
    where T : IRayCastable
{
    public struct HitResult
    {
        public T primitive;
        public RayHitResult rayHitResult;
    }

    // Slang buffer of BVHNode copied from Python.
    public RWStructuredBuffer<BVHNode> nodes;
    // Maximum number of nodes in the BVH, in this case, it'll be the size of the nodes buffer.
    public uint maxNodes;
    // Slang buffer of primitives represented by the BVH.
    public RWStructuredBuffer<T> primitives;
    // Number of primitives in the primitives buffer.
    public uint numPrimitives;

    public __init(RWStructuredBuffer<BVHNode> nodes, RWStructuredBuffer<T> primitives, uint numPrimitives, uint maxNodes)
    {
        this.nodes = nodes;
        this.primitives = primitives;
        this.numPrimitives = numPrimitives;
        this.maxNodes = maxNodes;
    }

    struct HitStack<int N>
    {
        uint[N] nodeIndices;
        uint top;

        __init()
        {
            this.top = 0;
        }

        [mutating]
        void push(uint nodeIndex)
        {
            this.nodeIndices[this.top] = nodeIndex;
            this.top += 1;
        }

        [mutating]
        uint pop()
        {
            this.top -= 1;
            return this.nodeIndices[this.top];
        }

        bool isEmpty()
        {
            return this.top == 0;
        }

        uint size()
        {
            return this.top;
        }
    }

    // BVH traversal to find the closest hit.
    public Optional<HitResult> hit(Ray ray)
    {
        // Initialize traversal.
        Optional<HitResult> hitResult = Optional<HitResult>();
        HitStack<64> stack;

        // TODO: Student implementation starts here.
        stack.push(0);
        uint currentIndex;
        BVHNode currentNode;
        T primitive;
        Optional<RayHitResult> rayHitResult;
        Optional<RayHitResult> closestRayHitResult;
        T closestPrimitive;
    

        while (stack.size() > 0) {
            currentIndex = stack.pop();
            currentNode = nodes[currentIndex];
            
            // Check if this is a leaf node
            if (currentNode.left == -1 && currentNode.right == -1) {
                for (uint i = currentNode.primLeft; i < currentNode.primRight; i++) {
                    rayHitResult = primitives[i].hit(ray);
                    if (rayHitResult.hasValue) {
                        if (!closestRayHitResult.hasValue || rayHitResult.value.t < closestRayHitResult.value.t) {
                            closestRayHitResult = rayHitResult;
                            closestPrimitive = primitives[i];
                        }
                    }
                }
            }
            else {
                // Internal node
                float2 leftNodeHit = nodes[currentNode.left].bound.hit(ray);
                float2 rightNodeHit = nodes[currentNode.right].bound.hit(ray);

                bool leftHits = (leftNodeHit.x != -1.0);
                bool rightHits = (rightNodeHit.x != -1.0);

                if (leftHits && rightHits) {
                    uint first = (leftNodeHit.x <= rightNodeHit.x) ? currentNode.left : currentNode.right;
                    uint second = (leftNodeHit.x <= rightNodeHit.x) ? currentNode.right : currentNode.left;
                    stack.push(second);
                    stack.push(first);
                } else if (leftHits) {
                    stack.push(currentNode.left);
                } else if (rightHits) {
                    stack.push(currentNode.right);
                }
            }
        }

        if (closestRayHitResult.hasValue) {
            hitResult = Optional<HitResult>(HitResult(closestPrimitive, closestRayHitResult.value));
        }
        // TODO: Student implementation ends here.

        return hitResult;
    }
}
