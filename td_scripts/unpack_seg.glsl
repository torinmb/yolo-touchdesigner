// Uniforms (if you need to control max layers dynamically)
// uniform int uMaxLayers;

layout (local_size_x = 8, local_size_y = 8) in;
void main()
{
    // 1. Get pixel coordinates
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    ivec3 outSize = imageSize(mTDComputeOutputs[0]);

    if (coord.x >= outSize.x || coord.y >= outSize.y) return;

    // 2. Fetch Interpolated Value (Smart Smooth)
    vec2 inRes = vec2(uTD2DInfos[0].res.zw); // In TouchDesigner, res.zw is (width, height), res.xy is (1/w, 1/h)
    vec2 outRes = vec2(outSize.xy);
    vec2 uv = (vec2(coord) + 0.5) / outRes;
    vec2 inPos = uv * inRes - 0.5;

    ivec2 tl = ivec2(floor(inPos));
    vec2 f = fract(inPos);
    
    // Bicubic-like smoothing (Optional but helps remove strair-stepping)
    f = f * f * (3.0 - 2.0 * f);

    // Fetch 4 Neighbors
    float v00 = texelFetch(sTD2DInputs[0], clamp(tl + ivec2(0, 0), ivec2(0), ivec2(inRes)-1), 0).r;
    float v10 = texelFetch(sTD2DInputs[0], clamp(tl + ivec2(1, 0), ivec2(0), ivec2(inRes)-1), 0).r;
    float v01 = texelFetch(sTD2DInputs[0], clamp(tl + ivec2(0, 1), ivec2(0), ivec2(inRes)-1), 0).r;
    float v11 = texelFetch(sTD2DInputs[0], clamp(tl + ivec2(1, 1), ivec2(0), ivec2(inRes)-1), 0).r;

    // -------------------------------------------------------------------
    // DECODING LOGIC (ID.Alpha)
    // -------------------------------------------------------------------
    
    // We can't linearly interpolate the PACKED value (e.g. 5.9 mixing with 0.0 gives 2.9 -> Wrong Class!)
    // So we must decode each neighbor FIRST, then decide what to do.
    
    // Unpack Neighbor 00
    int id00 = int(floor(v00)) - 1;
    float a00 = fract(v00) / 0.99;

    // Unpack Neighbor 10
    int id10 = int(floor(v10)) - 1;
    float a10 = fract(v10) / 0.99;

    // Unpack Neighbor 01
    int id01 = int(floor(v01)) - 1;
    float a01 = fract(v01) / 0.99;

    // Unpack Neighbor 11
    int id11 = int(floor(v11)) - 1;
    float a11 = fract(v11) / 0.99;

    // -------------------------------------------------------------------
    // LAYER LOOP
    // -------------------------------------------------------------------
    for (int i = 0; i < outSize.z; i++)
    {
        // For THIS layer 'i', check if neighbor belongs to it.
        // If yes, use its alpha. If no, alpha is 0.0.
        float alpha00 = (id00 == i) ? a00 : 0.0;
        float alpha10 = (id10 == i) ? a10 : 0.0;
        float alpha01 = (id01 == i) ? a01 : 0.0;
        float alpha11 = (id11 == i) ? a11 : 0.0;

        // NOW we can interpolate the alphas safely!
        float top = mix(alpha00, alpha10, f.x);
        float bot = mix(alpha01, alpha11, f.x);
        float finalAlpha = mix(top, bot, f.y);

        // Optional: boost contrast slightly if it feels too soft
        // finalAlpha = smoothstep(0.05, 0.95, finalAlpha);

        imageStore(mTDComputeOutputs[0], ivec3(coord, i), TDOutputSwizzle(vec4(finalAlpha)));
    }
}