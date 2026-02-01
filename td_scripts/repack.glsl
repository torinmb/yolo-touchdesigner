layout(local_size_x = 16, local_size_y = 16) in;
uniform float bypass;

void main() {
	if (bypass == 1.0) return;
	ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
	
    
    ivec2 inSize = textureSize(sTD2DInputs[0], 0);     // (W, H)
    int W = inSize.x;
    int H = inSize.y;

    // Our output is (3W, H). Guard threads outside H.
    if (gid.y >= H || gid.x >= 3*W) return;

    // Decide which plane we’re writing based on x:
    // [0..W-1] -> R plane, [W..2W-1] -> G plane, [2W..3W-1] -> B plane
    int plane = gid.x / W;          // 0,1,2
    int xIn   = gid.x - plane * W;  // 0..W-1
    ivec2 inUV = ivec2(xIn, gid.y);

    vec3 rgb = texelFetch(sTD2DInputs[0], inUV, 0).rgb;

    // Quantize to 0..1 UNORM (Compute TOP will store 8-bit if output is 8-bit fixed)
    float v = (plane == 0) ? rgb.r : (plane == 1) ? rgb.g : rgb.b;

    // If output is **mono**, only .r matters; if RGBA8, the helper swizzle will map it.
    imageStore(mTDComputeOutputs[0], gid, TDOutputSwizzle(vec4(v, 0.0, 0.0, 1.0)));
}
