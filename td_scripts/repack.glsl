layout(local_size_x = 16, local_size_y = 16) in;
uniform float bypass;

void main() {
	if (bypass == 1.0) return;
	ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
	
    ivec2 inSize = textureSize(sTD2DInputs[0], 0);     // (W, H)
    int W = inSize.x;
    int H = inSize.y;

    // Our output is (W, 3*H). Guard threads outside W and 3*H.
    if (gid.x >= W || gid.y >= 3*H) return;

    // Decide which plane we’re writing based on y:
    // [0..H-1] -> R plane, [H..2H-1] -> G plane, [2H..3H-1] -> B plane
    int plane = gid.y / H;          // 0,1,2
    int yIn   = gid.y - plane * H;  // 0..H-1
    ivec2 inUV = ivec2(gid.x, yIn);

    vec3 rgb = texelFetch(sTD2DInputs[0], inUV, 0).rgb;

    // Quantize to 0..1 UNORM (Compute TOP will store 8-bit if output is 8-bit fixed)
    float v = (plane == 0) ? rgb.r : (plane == 1) ? rgb.g : rgb.b;

    // If output is mono, only .r matters; if RGBA8, the helper swizzle will map it.
    imageStore(mTDComputeOutputs[0], gid, TDOutputSwizzle(vec4(v, 0.0, 0.0, 1.0)));
}
