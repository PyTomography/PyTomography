/* PyTomography image viewer (MIT licence). Colormap tables from matplotlib (matplotlib licence).
 *
 * Draws a tutorial's images on a 2D canvas the way matplotlib's imshow does with `extent`: every image stays on
 * its own voxel grid, placed at its position in scanner coordinates, and the layers are overlaid on screen.
 * Nothing is resampled onto the CT. SPECT and PET use imshow's Gaussian interpolation (sigma = half a voxel, on
 * the values, before the colormap); CT, MR and attenuation maps are bilinear; "Smooth display" off shows the voxels.
 * The 3D view is a rotating maximum-intensity projection of the SPECT or PET over a line integral of the CT.
 *
 *   const v = await PTViewer.mount(element, {manifest: "https://.../t_dicomdata/manifest.json"});
 *   v.destroy();   // stops the workers and frees the images
 *
 * The manifest is written by docs/tools/viewer_export.py; its layer files are NIfTI-1 (.nii.gz), axis-aligned.
 */
(function () {
  'use strict';
  if (window.PTViewer) return;

  const CMAPS = {"inferno":"00000401000501010601010802010a02020c02020e03021004031204031405041706041907051b08051d09061f0a07220b07240c08260d08290e092b10092d110a30120a32140b34150b37160b39180c3c190c3e1b0c411c0c431e0c451f0c48210c4a230c4c240c4f260c51280b53290b552b0b572d0b592f0a5b310a5c320a5e340a5f3609613809623909633b09643d09653e0966400a67420a68440a68450a69470b6a490b6a4a0c6b4c0c6b4d0d6c4f0d6c510e6c520e6d540f6d550f6d57106e59106e5a116e5c126e5d126e5f136e61136e62146e64156e65156e67166e69166e6a176e6c186e6d186e6f196e71196e721a6e741a6e751b6e771c6d781c6d7a1d6d7c1d6d7d1e6d7f1e6c801f6c82206c84206b85216b87216b88226a8a226a8c23698d23698f24699025689225689326679526679727669827669a28659b29649d29649f2a63a02a63a22b62a32c61a52c60a62d60a82e5fa92e5eab2f5ead305dae305cb0315bb1325ab3325ab43359b63458b73557b93556ba3655bc3754bd3853bf3952c03a51c13a50c33b4fc43c4ec63d4dc73e4cc83f4bca404acb4149cc4248ce4347cf4446d04545d24644d34743d44842d54a41d74b3fd84c3ed94d3dda4e3cdb503bdd513ade5238df5337e05536e15635e25734e35933e45a31e55c30e65d2fe75e2ee8602de9612bea632aeb6429eb6628ec6726ed6925ee6a24ef6c23ef6e21f06f20f1711ff1731df2741cf3761bf37819f47918f57b17f57d15f67e14f68013f78212f78410f8850ff8870ef8890cf98b0bf98c0af98e09fa9008fa9207fa9407fb9606fb9706fb9906fb9b06fb9d07fc9f07fca108fca309fca50afca60cfca80dfcaa0ffcac11fcae12fcb014fcb216fcb418fbb61afbb81dfbba1ffbbc21fbbe23fac026fac228fac42afac62df9c72ff9c932f9cb35f8cd37f8cf3af7d13df7d340f6d543f6d746f5d949f5db4cf4dd4ff4df53f4e156f3e35af3e55df2e661f2e865f2ea69f1ec6df1ed71f1ef75f1f179f2f27df2f482f3f586f3f68af4f88ef5f992f6fa96f8fb9af9fc9dfafda1fcffa4","magma":"00000401000501010601010802010902020b02020d03030f03031204041405041606051806051a07061c08071e0907200a08220b09240c09260d0a290e0b2b100b2d110c2f120d31130d34140e36150e38160f3b180f3d19103f1a10421c10441d11471e114920114b21114e22115024125325125527125829115a2a115c2c115f2d11612f116331116533106734106936106b38106c390f6e3b0f703d0f713f0f72400f74420f75440f764510774710784910784a10794c117a4e117b4f127b51127c52137c54137d56147d57157e59157e5a167e5c167f5d177f5f187f601880621980641a80651a80671b80681c816a1c816b1d816d1d816e1e81701f81721f817320817521817621817822817922827b23827c23827e24828025828125818326818426818627818827818928818b29818c29818e2a81902a81912b81932b80942c80962c80982d80992d809b2e7f9c2e7f9e2f7fa02f7fa1307ea3307ea5317ea6317da8327daa337dab337cad347cae347bb0357bb2357bb3367ab5367ab73779b83779ba3878bc3978bd3977bf3a77c03a76c23b75c43c75c53c74c73d73c83e73ca3e72cc3f71cd4071cf4070d0416fd2426fd3436ed5446dd6456cd8456cd9466bdb476adc4869de4968df4a68e04c67e24d66e34e65e44f64e55064e75263e85362e95462ea5661eb5760ec5860ed5a5fee5b5eef5d5ef05f5ef1605df2625df2645cf3655cf4675cf4695cf56b5cf66c5cf66e5cf7705cf7725cf8745cf8765cf9785df9795df97b5dfa7d5efa7f5efa815ffb835ffb8560fb8761fc8961fc8a62fc8c63fc8e64fc9065fd9266fd9467fd9668fd9869fd9a6afd9b6bfe9d6cfe9f6dfea16efea36ffea571fea772fea973feaa74feac76feae77feb078feb27afeb47bfeb67cfeb77efeb97ffebb81febd82febf84fec185fec287fec488fec68afec88cfeca8dfecc8ffecd90fecf92fed194fed395fed597fed799fed89afdda9cfddc9efddea0fde0a1fde2a3fde3a5fde5a7fde7a9fde9aafdebacfcecaefceeb0fcf0b2fcf2b4fcf4b6fcf6b8fcf7b9fcf9bbfcfbbdfcfdbf","plasma":"0d088710078813078916078a19068c1b068d1d068e20068f2206902406912605912805922a05932c05942e05952f059631059733059735049837049938049a3a049a3c049b3e049c3f049c41049d43039e44039e46039f48039f4903a04b03a14c02a14e02a25002a25102a35302a35502a45601a45801a45901a55b01a55c01a65e01a66001a66100a76300a76400a76600a76700a86900a86a00a86c00a86e00a86f00a87100a87201a87401a87501a87701a87801a87a02a87b02a87d03a87e03a88004a88104a78305a78405a78606a68707a68808a68a09a58b0aa58d0ba58e0ca48f0da4910ea3920fa39410a29511a19613a19814a099159f9a169f9c179e9d189d9e199da01a9ca11b9ba21d9aa31e9aa51f99a62098a72197a82296aa2395ab2494ac2694ad2793ae2892b02991b12a90b22b8fb32c8eb42e8db52f8cb6308bb7318ab83289ba3388bb3488bc3587bd3786be3885bf3984c03a83c13b82c23c81c33d80c43e7fc5407ec6417dc7427cc8437bc9447aca457acb4679cc4778cc4977cd4a76ce4b75cf4c74d04d73d14e72d24f71d35171d45270d5536fd5546ed6556dd7566cd8576bd9586ada5a6ada5b69db5c68dc5d67dd5e66de5f65de6164df6263e06363e16462e26561e26660e3685fe4695ee56a5de56b5de66c5ce76e5be76f5ae87059e97158e97257ea7457eb7556eb7655ec7754ed7953ed7a52ee7b51ef7c51ef7e50f07f4ff0804ef1814df1834cf2844bf3854bf3874af48849f48948f58b47f58c46f68d45f68f44f79044f79143f79342f89441f89540f9973ff9983ef99a3efa9b3dfa9c3cfa9e3bfb9f3afba139fba238fca338fca537fca636fca835fca934fdab33fdac33fdae32fdaf31fdb130fdb22ffdb42ffdb52efeb72dfeb82cfeba2cfebb2bfebd2afebe2afec029fdc229fdc328fdc527fdc627fdc827fdca26fdcb26fccd25fcce25fcd025fcd225fbd324fbd524fbd724fad824fada24f9dc24f9dd25f8df25f8e125f7e225f7e425f6e626f6e826f5e926f5eb27f4ed27f3ee27f3f027f2f227f1f426f1f525f0f724f0f921","viridis":"44015444025645045745055946075a46085c460a5d460b5e470d60470e6147106347116447136548146748166848176948186a481a6c481b6d481c6e481d6f481f70482071482173482374482475482576482677482878482979472a7a472c7a472d7b472e7c472f7d46307e46327e46337f463480453581453781453882443983443a83443b84433d84433e85423f854240864241864142874144874045884046883f47883f48893e49893e4a893e4c8a3d4d8a3d4e8a3c4f8a3c508b3b518b3b528b3a538b3a548c39558c39568c38588c38598c375a8c375b8d365c8d365d8d355e8d355f8d34608d34618d33628d33638d32648e32658e31668e31678e31688e30698e306a8e2f6b8e2f6c8e2e6d8e2e6e8e2e6f8e2d708e2d718e2c718e2c728e2c738e2b748e2b758e2a768e2a778e2a788e29798e297a8e297b8e287c8e287d8e277e8e277f8e27808e26818e26828e26828e25838e25848e25858e24868e24878e23888e23898e238a8d228b8d228c8d228d8d218e8d218f8d21908d21918c20928c20928c20938c1f948c1f958b1f968b1f978b1f988b1f998a1f9a8a1e9b8a1e9c891e9d891f9e891f9f881fa0881fa1881fa1871fa28720a38620a48621a58521a68522a78522a88423a98324aa8325ab8225ac8226ad8127ad8128ae8029af7f2ab07f2cb17e2db27d2eb37c2fb47c31b57b32b67a34b67935b77937b87838b9773aba763bbb753dbc743fbc7340bd7242be7144bf7046c06f48c16e4ac16d4cc26c4ec36b50c46a52c56954c56856c66758c7655ac8645cc8635ec96260ca6063cb5f65cb5e67cc5c69cd5b6ccd5a6ece5870cf5773d05675d05477d1537ad1517cd2507fd34e81d34d84d44b86d54989d5488bd6468ed64590d74393d74195d84098d83e9bd93c9dd93ba0da39a2da37a5db36a8db34aadc32addc30b0dd2fb2dd2db5de2bb8de29bade28bddf26c0df25c2df23c5e021c8e020cae11fcde11dd0e11cd2e21bd5e21ad8e219dae319dde318dfe318e2e418e5e419e7e419eae51aece51befe51cf1e51df4e61ef6e620f8e621fbe723fde725","hot":"0b00000d00001000001200001500001800001a00001d00002000002200002500002700002a00002d00002f00003200003500003700003a00003c00003f00004200004400004700004a00004c00004f00005100005400005700005900005c00005f00006100006400006600006900006c00006e00007100007400007600007900007b00007e00008100008300008600008900008b00008e00009000009300009600009800009b00009e0000a00000a30000a50000a80000ab0000ad0000b00000b30000b50000b80000ba0000bd0000c00000c20000c50000c80000ca0000cd0000cf0000d20000d50000d70000da0000dd0000df0000e20000e40000e70000ea0000ec0000ef0000f20000f40000f70000f90000fc0000ff0000ff0200ff0500ff0800ff0a00ff0d00ff1000ff1200ff1500ff1700ff1a00ff1d00ff1f00ff2200ff2500ff2700ff2a00ff2c00ff2f00ff3200ff3400ff3700ff3a00ff3c00ff3f00ff4100ff4400ff4700ff4900ff4c00ff4f00ff5100ff5400ff5600ff5900ff5c00ff5e00ff6100ff6400ff6600ff6900ff6b00ff6e00ff7100ff7300ff7600ff7900ff7b00ff7e00ff8000ff8300ff8600ff8800ff8b00ff8e00ff9000ff9300ff9500ff9800ff9b00ff9d00ffa000ffa200ffa500ffa800ffaa00ffad00ffb000ffb200ffb500ffb700ffba00ffbd00ffbf00ffc200ffc500ffc700ffca00ffcc00ffcf00ffd200ffd400ffd700ffda00ffdc00ffdf00ffe100ffe400ffe700ffe900ffec00ffef00fff100fff400fff600fff900fffc00fffe00ffff03ffff07ffff0bffff0fffff13ffff17ffff1bffff1fffff22ffff26ffff2affff2effff32ffff36ffff3affff3effff42ffff46ffff4affff4effff52ffff56ffff5affff5effff61ffff65ffff69ffff6dffff71ffff75ffff79ffff7dffff81ffff85ffff89ffff8dffff91ffff95ffff99ffff9dffffa0ffffa4ffffa8ffffacffffb0ffffb4ffffb8ffffbcffffc0ffffc4ffffc8ffffccffffd0ffffd4ffffd8ffffdcffffdfffffe3ffffe7ffffebffffeffffff3fffff7fffffbffffff","gist_heat":"0000000200000300000400000600000800000900000a00000c00000e00000f00001000001200001400001500001600001800001a00001b00001c00001e00002000002100002200002400002600002700002800002a00002c00002d00002e00003000003100003300003500003600003700003900003b00003c00003d00003f00004000004200004300004500004600004800004a00004b00004d00004e00005000005100005200005400005600005700005800005a00005b00005d00005e00006000006200006300006500006600006800006900006a00006c00006e00006f00007000007200007300007500007600007800007a00007b00007d00007e00008000008100008200008400008500008700008800008a00008b00008d00008e00009000009200009300009400009600009800009900009b00009c00009e00009f0000a00000a20000a30000a50000a60000a80000aa0000ab0000ac0000ae0000b00000b10000b20000b40000b50000b70000b80000ba0000bb0000bd0000be0000c00100c20300c30500c40700c60900c80b00c90d00cb0f00cc1100ce1300cf1500d01700d21900d41b00d51d00d61f00d82100da2300db2500dc2700de2900e02b00e12d00e22f00e43100e53300e73500e83700ea3900ec3b00ed3d00ee3f00f04100f24300f34500f44700f64900f84b00f94d00fb4f00fc5100fe5300ff5500ff5700ff5900ff5b00ff5d00ff5f00ff6100ff6300ff6500ff6700ff6900ff6b00ff6d00ff6f00ff7100ff7300ff7500ff7700ff7900ff7b00ff7d00ff7f00ff8103ff8307ff850bff870fff8913ff8b17ff8d1bff8f1fff9123ff9327ff952bff972fff9933ff9b37ff9d3bff9f3fffa143ffa347ffa54bffa74fffa953ffab57ffad5bffaf5fffb163ffb367ffb56bffb76fffb973ffbb77ffbd7bffbf7fffc183ffc387ffc58bffc78fffc993ffcb97ffcd9bffcf9fffd1a3ffd3a7ffd5abffd7afffd9b3ffdbb7ffddbbffdfbfffe1c3ffe3c7ffe5cbffe7cfffe9d3ffebd7ffeddbffefdffff1e3fff3e7fff5ebfff7effff9f3fffbf7fffdfbffffff","nipy_spectral":"00000009000b1300151c002025002b2f003538004041004b4b00555400605d006b67007570008077008879008a7a008b7b008c7d008e7e008f7f009081009282009383009485009686009787009883009a78009b6d009c63009e58009f4d00a04300a23800a32d00a42300a61800a70d00a80300aa0000ad0000b10000b50000b90000bd0000c10000c50000c90000cd0000d10000d50000d90000dd0009dd0013dd001cdd0025dd002fdd0038dd0041dd004bdd0054dd005ddd0067dd0070dd0078dd007add007ddd0080dd0082dd0085dd0088dd008add008ddd0090dd0092dd0095dd0098dd009adb009bd7009cd3009ecf009fcb00a0c700a2c300a3bf00a4bb00a6b700a7b300a8af00aaab00aaa800aaa500aaa300aaa000aa9d00aa9b00aa9800aa9500aa9300aa9000aa8d00aa8b00aa8800a97d00a77300a66800a55d00a35300a24800a13d009f33009e28009d1d009b13009a08009a00009c00009f0000a20000a40000a70000aa0000ac0000af0000b20000b40000b70000ba0000bc0000bf0000c20000c40000c70000ca0000cc0000cf0000d20000d40000d70000da0000dc0000df0000e20000e40000e70000ea0000ec0000ef0000f20000f40000f70000fa0000fc0000ff000fff001dff002cff003bff0049ff0058ff0067ff0075ff0084ff0093ff00a1ff00b0ff00bcff00c0fd00c4fc00c8fb00ccf900d0f800d4f700d8f500dcf400e0f300e4f100e8f000ecef00efed00f0ea00f1e700f3e500f4e200f5df00f7dd00f8da00f9d700fbd500fcd200fdcf00ffcd00ffc900ffc500ffc100ffbd00ffb900ffb500ffb100ffad00ffa900ffa500ffa100ff9d00ff9900ff8d00ff8100ff7500ff6900ff5d00ff5100ff4500ff3900ff2d00ff2100ff1500ff0900fe0000fc0000f90000f60000f40000f10000ee0000ec0000e90000e60000e40000e10000de0000dc0000db0000da0000d80000d70000d60000d40000d30000d20000d00000cf0000ce0000cc0000cc0c0ccc1c1ccc2c2ccc3c3ccc4c4ccc5c5ccc6c6ccc7c7ccc8c8ccc9c9cccacacccbcbccccccc","gray":"0000000101010202020303030404040505050606060707070808080909090a0a0a0b0b0b0c0c0c0d0d0d0e0e0e0f0f0f1010101111111212121313131414141515151616161717171818181919191a1a1a1b1b1b1c1c1c1d1d1d1e1e1e1f1f1f2020202121212222222323232424242525252626262727272828282929292a2a2a2b2b2b2c2c2c2d2d2d2e2e2e2f2f2f3030303131313232323333333434343535353636363737373838383939393a3a3a3b3b3b3c3c3c3d3d3d3e3e3e3f3f3f4040404141414242424343434444444545454646464747474848484949494a4a4a4b4b4b4c4c4c4d4d4d4e4e4e4f4f4f5050505151515252525353535454545555555656565757575858585959595a5a5a5b5b5b5c5c5c5d5d5d5e5e5e5f5f5f6060606161616262626363636464646565656666666767676868686969696a6a6a6b6b6b6c6c6c6d6d6d6e6e6e6f6f6f7070707171717272727373737474747575757676767777777878787979797a7a7a7b7b7b7c7c7c7d7d7d7e7e7e7f7f7f8080808181818282828383838484848585858686868787878888888989898a8a8a8b8b8b8c8c8c8d8d8d8e8e8e8f8f8f9090909191919292929393939494949595959696969797979898989999999a9a9a9b9b9b9c9c9c9d9d9d9e9e9e9f9f9fa0a0a0a1a1a1a2a2a2a3a3a3a4a4a4a5a5a5a6a6a6a7a7a7a8a8a8a9a9a9aaaaaaabababacacacadadadaeaeaeafafafb0b0b0b1b1b1b2b2b2b3b3b3b4b4b4b5b5b5b6b6b6b7b7b7b8b8b8b9b9b9babababbbbbbbcbcbcbdbdbdbebebebfbfbfc0c0c0c1c1c1c2c2c2c3c3c3c4c4c4c5c5c5c6c6c6c7c7c7c8c8c8c9c9c9cacacacbcbcbcccccccdcdcdcecececfcfcfd0d0d0d1d1d1d2d2d2d3d3d3d4d4d4d5d5d5d6d6d6d7d7d7d8d8d8d9d9d9dadadadbdbdbdcdcdcdddddddedededfdfdfe0e0e0e1e1e1e2e2e2e3e3e3e4e4e4e5e5e5e6e6e6e7e7e7e8e8e8e9e9e9eaeaeaebebebecececedededeeeeeeefefeff0f0f0f1f1f1f2f2f2f3f3f3f4f4f4f5f5f5f6f6f6f7f7f7f8f8f8f9f9f9fafafafbfbfbfcfcfcfdfdfdfefefeffffff"};
  const LUT = {};
  for (const [k, hex] of Object.entries(CMAPS)) {
    const a = new Uint8Array(768);
    for (let i = 0; i < 768; i++) a[i] = parseInt(hex.substr(2 * i, 2), 16);
    LUT[k] = a;
  }
  const CMAP_ORDER = ['inferno', 'magma', 'plasma', 'viridis', 'hot', 'gist_heat', 'nipy_spectral', 'gray'].filter(k => LUT[k]);
  const CT_WINDOWS = [['phantom', 'Phantom', -50, 300], ['soft', 'Soft tissue', -160, 240], ['bone', 'Bone', -450, 1050],
    ['lung', 'Lung', -1350, 150], ['full', 'Full range', -1000, 1000]];
  // SPECT and PET are drawn in colour over a grey anatomical image (CT, MR or attenuation map)
  const ROLE = {spect: 'overlay', pet: 'overlay', ct: 'base', mr: 'base', mu: 'base', image: 'base'};
  // the colour images' first 3% above the lower limit fade in from clear (see rgbaOf)
  const FADE = 0.03;
  // u = screen right, v = screen down, each [scanner axis, sign]; n = the axis through the slice. Radiological, as
  // imshow shows PyTomography's arrays: the patient's right on the left, anterior and superior at the top.
  const VIEWS = {
    axial: {u: [0, -1], v: [1, -1], n: 2, o: ['R', 'L', 'A', 'P'], label: 'Axial'},
    coronal: {u: [0, -1], v: [2, -1], n: 1, o: ['R', 'L', 'S', 'I'], label: 'Coronal'},
    sagittal: {u: [1, -1], v: [2, -1], n: 0, o: ['A', 'P', 'S', 'I'], label: 'Sagittal'},
  };
  const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
  const sortPair = p => p[0] <= p[1] ? p : [p[1], p[0]];
  const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
  const fmtN = v => !Number.isFinite(v) ? '–' : Math.abs(v) >= 1000 ? v.toFixed(0) : Math.abs(v) >= 100 ? v.toFixed(0) :
    Math.abs(v) >= 10 ? v.toFixed(1) : Math.abs(v) >= 0.01 || v === 0 ? v.toFixed(2) : v.toExponential(2);
  const fmtMB = b => (b / 1e6).toFixed(b < 1e7 ? 1 : 0) + ' MB';

  // ---------- files ----------
  async function fetchBytes(url, onBytes, signal) {
    const r = await fetch(url, {signal});
    if (!r.ok) throw new Error(`${url} answered ${r.status}`);
    let u8;
    if (onBytes && r.body && r.body.getReader) {
      const rd = r.body.getReader(), parts = [];
      let n = 0;
      for (;;) {
        const {done, value} = await rd.read();
        if (done) break;
        parts.push(value); n += value.length; onBytes(n);
      }
      u8 = new Uint8Array(n);
      let o = 0;
      for (const p of parts) { u8.set(p, o); o += p.length; }
    } else u8 = new Uint8Array(await r.arrayBuffer());
    if (u8[0] === 0x7b) {  // {"base64": ...}: for hosts that can't serve .nii.gz (the prototype's)
      const s = JSON.parse(new TextDecoder().decode(u8)).base64, b = atob(s);
      u8 = new Uint8Array(b.length);
      for (let i = 0; i < b.length; i++) u8[i] = b.charCodeAt(i);
    }
    return u8;
  }
  async function gunzip(u8) {
    if (typeof DecompressionStream === 'undefined')
      throw new Error('this browser cannot unpack the images; Safari 16.4, Chrome 80, Firefox 113 or later can');
    return new Uint8Array(await new Response(new Blob([u8]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
  }
  async function readNifti(u8, name) {
    if (u8[0] === 0x1f && u8[1] === 0x8b) u8 = await gunzip(u8);
    const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    if (dv.getInt32(0, true) !== 348) throw new Error(`${name} is not a little-endian NIfTI-1 file`);
    const dims = [1, 2, 3].map(i => Math.max(1, dv.getInt16(40 + 2 * i, true))), dtype = dv.getInt16(70, true);
    const off = Math.round(dv.getFloat32(108, true));
    let sl = dv.getFloat32(112, true), it = dv.getFloat32(116, true);
    if (!Number.isFinite(sl) || sl === 0) { sl = 1; it = 0; }
    if (!Number.isFinite(it)) it = 0;
    if (dv.getInt16(254, true) <= 0) throw new Error(`${name} has no scanner coordinates (sform)`);
    const aff = [0, 1, 2].map(r => [0, 1, 2, 3].map(c => dv.getFloat32(280 + 16 * r + 4 * c, true)));
    aff.push([0, 0, 0, 1]);
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++)
      if (a !== b && Math.abs(aff[a][b]) > 1e-4 * Math.abs(aff[a][a])) throw new Error(`${name} is rotated against the scanner axes`);
    const T = {2: Uint8Array, 4: Int16Array, 8: Int32Array, 16: Float32Array, 64: Float64Array, 256: Int8Array, 512: Uint16Array, 768: Uint32Array}[dtype];
    if (!T) throw new Error(`${name} uses NIfTI data type ${dtype}, which the viewer does not read`);
    const n = dims[0] * dims[1] * dims[2];
    const raw = new T(u8.buffer.slice(u8.byteOffset + off, u8.byteOffset + off + n * T.BYTES_PER_ELEMENT));
    const img = new Float32Array(n);
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < n; i++) { const v = raw[i] * sl + it; img[i] = v; if (v < mn) mn = v; if (v > mx) mx = v; }
    return {img, dims, aff, pd: [0, 1, 2].map(a => Math.abs(aff[a][a])), min: mn, max: mx};
  }

  // ---------- matplotlib's "gaussian" interpolation: weights exp(-2 d^2) within 2 source pixels (sigma = 0.5 pixel) ----------
  const WCACHE = new Map();
  function gweights(n, f) {
    const key = n + ':' + f;
    if (WCACHE.has(key)) return WCACHE.get(key);
    const W = [];
    for (let o = 0; o < n * f; o++) {
      const u = (o + 0.5) / f - 0.5, ix = [], wv = [];
      let sum = 0;
      for (let s = Math.max(0, Math.ceil(u - 2)); s <= Math.min(n - 1, Math.floor(u + 2)); s++) {
        const d = u - s, w = Math.exp(-2 * d * d); ix.push(s); wv.push(w); sum += w;
      }
      for (let i = 0; i < wv.length; i++) wv[i] /= sum;
      W.push([Int32Array.from(ix), Float32Array.from(wv)]);
    }
    if (WCACHE.size > 64) WCACHE.clear();
    WCACHE.set(key, W);
    return W;
  }
  function gaussUp(sl, f) {
    const {w, h, data} = sl, W = w * f, H = h * f, wx = gweights(w, f), wy = gweights(h, f);
    const t = new Float32Array(W * h), out = new Float32Array(W * H);
    for (let r = 0; r < h; r++) {
      const so = r * w, to = r * W;
      for (let c = 0; c < W; c++) { const [ix, wv] = wx[c]; let acc = 0; for (let i = 0; i < ix.length; i++) acc += wv[i] * data[so + ix[i]]; t[to + c] = acc; }
    }
    for (let r = 0; r < H; r++) {
      const [iy, wv] = wy[r], to = r * W;
      for (let i = 0; i < iy.length; i++) { const so = iy[i] * W, ww = wv[i]; for (let c = 0; c < W; c++) out[to + c] += ww * t[so + c]; }
    }
    return {w: W, h: H, data: out, U: sl.U, V: sl.V};
  }

  // ---------- background workers ----------
  // Smoothing: a separable 3D Gaussian with sigma per axis in voxels, always from the unsmoothed data
  const SMOOTH_SRC = `self.onmessage=e=>{const{tag,data,dims,sig}=e.data;let a=data,b=new Float32Array(a.length);const[nx,ny,nz]=dims;
    const lines=[[ny,nz,nx,nx*ny,1],[nx,nz,1,nx*ny,nx],[nx,ny,1,nx,nx*ny]];
    for(let ax=0;ax<3;ax++){const s=sig[ax];if(!(s>0.05))continue;const r=Math.max(1,Math.ceil(3*s)),k=new Float32Array(2*r+1);let sum=0;
      for(let t=-r;t<=r;t++){const w=Math.exp(-0.5*(t/s)*(t/s));k[t+r]=w;sum+=w;}for(let t=0;t<k.length;t++)k[t]/=sum;
      const[n1,n2,s1,s2,st]=lines[ax],n=dims[ax];
      for(let p=0;p<n2;p++)for(let q=0;q<n1;q++){const b0=p*s2+q*s1;
        for(let i=0;i<n;i++){let acc=0;for(let t=-r;t<=r;t++){let j=i+t;if(j<0)j=0;else if(j>=n)j=n-1;acc+=k[t+r]*a[b0+j*st];}b[b0+i*st]=acc;}}
      const tmp=a;a=b;b=tmp;}
    self.postMessage({tag,data:a},[a.buffer]);};`;
  // 3D view: the overlay's maximum intensity over a line integral of the base image (a DRR), on a coarse grid,
  // turned about the scanner's z axis. With no overlay it is the DRR alone; at 0 degrees it is an anterior view.
  // Under a colour image the DRR is drawn at about half brightness, and the colour's opacity rises with intensity
  // (opacity x sqrt of its place between the limits), so hot spots stand out and cold background stays clear.
  const MIP_SRC = `let G=null,ov=null,mu=null,dmax=1;
  function resample(img,dims,aff,map){const[nx,ny,nz]=dims,out=new Float32Array(G.gx*G.gy*G.gz);let n=0;
    for(let k=0;k<G.gz;k++){const fk=(G.z0+k*G.h-aff[2][3])/aff[2][2];for(let j=0;j<G.gy;j++){const fj=(G.y0+j*G.h-aff[1][3])/aff[1][1];
      for(let i=0;i<G.gx;i++,n++){const fi=(G.x0+i*G.h-aff[0][3])/aff[0][0];
        if(!(fi>=0&&fj>=0&&fk>=0&&fi<=nx-1&&fj<=ny-1&&fk<=nz-1)){out[n]=map(NaN);continue;}
        const i0=Math.floor(fi),j0=Math.floor(fj),k0=Math.floor(fk),i1=Math.min(i0+1,nx-1),j1=Math.min(j0+1,ny-1),k1=Math.min(k0+1,nz-1),u=fi-i0,v=fj-j0,w=fk-k0;
        const I=(a,b,c)=>img[a+nx*(b+ny*c)];
        const c0=(I(i0,j0,k0)*(1-u)+I(i1,j0,k0)*u)*(1-v)+(I(i0,j1,k0)*(1-u)+I(i1,j1,k0)*u)*v,c1=(I(i0,j0,k1)*(1-u)+I(i1,j0,k1)*u)*(1-v)+(I(i0,j1,k1)*(1-u)+I(i1,j1,k1)*u)*v;
        out[n]=map(c0*(1-w)+c1*w);}}}return out;}
  function density(kind,scale){return kind==='ct'?(v=>Number.isNaN(v)?0:Math.max(0,(v+1000)/1000)):(v=>Number.isNaN(v)?0:Math.max(0,v/scale));}
  function setMu(d){mu=resample(d.img,d.dims,d.aff,density(d.kind,d.scale||1));let m=0;
    for(let k=0;k<G.gz;k++)for(let j=0;j<G.gy;j++){let s=0;for(let i=0;i<G.gx;i++)s+=mu[i+G.gx*(j+G.gy*k)];if(s>m)m=s;}
    for(let k=0;k<G.gz;k++)for(let i=0;i<G.gx;i++){let s=0;for(let j=0;j<G.gy;j++)s+=mu[i+G.gx*(j+G.gy*k)];if(s>m)m=s;}dmax=Math.max(1e-6,m);}
  function render(p){const{gx,gy,gz}=G,cx=(gx-1)/2,cy=(gy-1)/2,W=Math.ceil(Math.hypot(gx,gy))+2,H=gz,c=Math.cos(p.th),s=Math.sin(p.th),hw=(W-1)/2,slab=gx*gy;
    const out=new Uint8ClampedArray(W*H*4),lut=p.lut,span=Math.max(1e-12,p.hi-p.lo),useMu=!!mu&&p.ctw>0,useOv=!!ov;
    for(let r=0;r<H;r++){const off=(gz-1-r)*slab;for(let u=0;u<W;u++){const uu=hw-u;let m=-Infinity,d=0;
      for(let t=0;t<W;t++){const ss=t-hw,x=cx+uu*c-ss*s,y=cy+uu*s+ss*c;if(x<0||y<0||x>gx-1||y>gy-1)continue;
        if(p.smooth){const x0=x|0,y0=y|0,x1=Math.min(x0+1,gx-1),y1=Math.min(y0+1,gy-1),fx=x-x0,fy=y-y0,a=off+x0+gx*y0,b=off+x1+gx*y0,e=off+x0+gx*y1,f=off+x1+gx*y1;
          if(useOv){const v=(ov[a]*(1-fx)+ov[b]*fx)*(1-fy)+(ov[e]*(1-fx)+ov[f]*fx)*fy;if(v>m)m=v;}
          if(useMu)d+=(mu[a]*(1-fx)+mu[b]*fx)*(1-fy)+(mu[e]*(1-fx)+mu[f]*fx)*fy;}
        else{const q=off+Math.round(x)+gx*Math.round(y);if(useOv&&ov[q]>m)m=ov[q];if(useMu)d+=mu[q];}}
      const g=useMu?Math.pow(Math.min(1,d/dmax),0.7)*255*p.ctw*(useOv?0.55:1):0,o=(r*W+u)*4;let R=g,Gc=g,B=g;
      if(useOv&&m>p.lo){const tt=Math.min(1,(m-p.lo)/span),li=Math.round(tt*255)*3,a=p.op*Math.sqrt(tt);R=(1-a)*R+a*lut[li];Gc=(1-a)*Gc+a*lut[li+1];B=(1-a)*B+a*lut[li+2];}
      out[o]=R;out[o+1]=Gc;out[o+2]=B;out[o+3]=255;}}
    return {W,H,rgba:out};}
  self.onmessage=e=>{const d=e.data;
    if(d.cmd==='grid'){G=d.grid;ov=null;mu=null;}
    else if(d.cmd==='base'){if(G)setMu(d);}
    else if(d.cmd==='nobase'){mu=null;}
    else if(d.cmd==='ov'){if(G)ov=resample(d.img,d.dims,d.aff,v=>Number.isNaN(v)?0:v);}
    else if(d.cmd==='noov'){ov=null;}
    else if(d.cmd==='render'&&G&&(ov||mu)){const f=render(d.p);self.postMessage({cmd:'frame',W:f.W,H:f.H,rgba:f.rgba,mm:G.h},[f.rgba.buffer]);}
    else if(d.cmd==='render')self.postMessage({cmd:'frame',W:0,H:0});};`;

  let NEXT = 0;

  async function mount(root, opt) {
    opt = opt || {};
    const uid = 'ptv' + (++NEXT), id = s => uid + '-' + s;
    const reduceMotion = !!(window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches);
    const coarse = !!(window.matchMedia && matchMedia('(pointer: coarse)').matches);
    const dpr = () => window.devicePixelRatio || 1;
    const cleanups = [];
    const on = (el, ev, fn, o) => { el.addEventListener(ev, fn, o); cleanups.push(() => el.removeEventListener(ev, fn, o)); };
    let alive = true, abort = typeof AbortController !== 'undefined' ? new AbortController() : null;

    const tiles = ['axial', 'coronal', 'sagittal'].map(v =>
      `<div class="ptv-tile" data-v="${v}"><canvas tabindex="0" aria-label="${VIEWS[v].label} slice; arrow keys move through the slices"></canvas>` +
      `<span class="ptv-tl">${VIEWS[v].label}</span><input type="range" class="ptv-slc" aria-label="${VIEWS[v].label} slice position"></div>`).join('');
    root.classList.add('ptv');
    root.dataset.mode = opt.maximized ? 'max' : 'inline';
    const hint = coarse ? 'Drag to move the crosshair, pinch to zoom and pan, and use the sliders to move through the slices. Drag the 3D view to turn it.'
      : '<kbd>Wheel</kbd> next slice · <kbd>Ctrl</kbd>+<kbd>Wheel</kbd> zoom · <kbd>Shift</kbd>+drag pan · click or drag to move the crosshair · drag the 3D view to turn it · arrow keys move through slices';
    root.innerHTML =
      `<div class="ptv-head"><div class="ptv-title"><b id="${id('title')}">${esc(opt.title || 'Loading…')}</b>` +
      `<span class="ptv-meta" id="${id('meta')}"><span id="${id('facts')}"></span><span class="ptv-credit" id="${id('credit')}"></span></span></div>` +
      `<div class="ptv-pick" id="${id('pickWrap')}" hidden><label for="${id('pick')}">Image</label><select id="${id('pick')}"></select></div>` +
      `<div class="ptv-actions"><button type="button" class="ptv-btn ptv-ctlbtn" id="${id('ctl')}" aria-expanded="false" aria-controls="${id('rail')}">Controls</button>` +
      `${opt.noMaximize ? '' : `<button type="button" class="ptv-btn" id="${id('max')}" aria-pressed="false">Full screen</button>`}` +
      `${opt.onClose ? `<button type="button" class="ptv-btn" id="${id('close')}">Close</button>` : ''}</div></div>` +
      `<div class="ptv-station"><div class="ptv-view">` +
      `<div class="ptv-bar"><div class="ptv-seg" role="group" aria-label="View" id="${id('views')}">` +
      [['multi', 'Slices + 3D'], ['axial', 'Axial'], ['coronal', 'Coronal'], ['sagittal', 'Sagittal'], ['mip', '3D']].map(([v, t]) =>
        `<button type="button" data-view="${v}" aria-pressed="${v === 'multi'}">${t}</button>`).join('') + `</div></div>` +
      `<div class="ptv-stage"><div class="ptv-tiles" id="${id('tiles')}" data-view="multi">${tiles}` +
      `<div class="ptv-tile" data-v="mip"><canvas id="${id('mip')}" tabindex="0" aria-label="Rotating 3D view; drag to turn it"></canvas><span class="ptv-tl" id="${id('miplab')}">3D</span>` +
      `<div class="ptv-mipbar"><button type="button" class="ptv-btn" id="${id('rot')}" aria-pressed="true">Rotating</button></div></div></div>` +
      `<div class="ptv-loading" id="${id('loading')}" role="status">Loading the images…</div></div>` +
      `<div class="ptv-status"><div class="ptv-cbar"><span id="${id('cbLo')}"></span><div class="ptv-grad" id="${id('cbGrad')}"></div><span id="${id('cbHi')}"></span></div>` +
      `<div class="ptv-read"><span id="${id('readout')}">–</span><span id="${id('where')}"></span></div></div></div>` +
      `<aside class="ptv-rail" id="${id('rail')}" aria-label="Viewer controls">` +
      `<div class="ptv-railhead"><b>Controls</b><button type="button" class="ptv-btn" id="${id('ctlDone')}">Done</button></div>` +
      `<div id="${id('cards')}" class="ptv-cards"></div>` +
      `<div class="ptv-row"><label class="ptv-check"><input type="checkbox" id="${id('interp')}" checked> Smooth display</label>` +
      `<button type="button" class="ptv-btn ptv-reset" id="${id('reset')}">Reset</button></div>` +
      `<details class="ptv-about"><summary>About the display and the controls</summary>` +
      `<p class="ptv-keys">${hint}</p>` +
      `<p class="ptv-note">Smooth display draws each SPECT or PET image as matplotlib's <code>interpolation="gaussian"</code> does (σ = half a voxel); ` +
      `CT is drawn bilinearly. Smoothing is a separate 3D Gaussian, with this full width at half maximum, applied to that image's data; the upper limit follows the smoothed image until you set it yourself. ` +
      `Values at or below the lower limit of the colour image are see-through.</p></details>` +
      `<p class="ptv-err" id="${id('err')}" role="status" aria-live="polite"></p></aside></div>`;
    const $ = s => document.getElementById(id(s));
    const err = m => { const e = $('err'); if (e) e.textContent = m || ''; };
    const setLoading = t => { const l = $('loading'); if (!l) return; l.hidden = !t; if (t) l.textContent = t; };

    const api = {
      el: root,
      destroy() {
        alive = false;
        if (abort) abort.abort();
        cleanups.forEach(f => { try { f(); } catch (_) {} });
        workers.forEach(w => w && w.terminate());
        urls.forEach(u => URL.revokeObjectURL(u));
        LAY = [];
        root.innerHTML = '';
        root.classList.remove('ptv');
      },
      reset: () => reset && reset(),
    };
    const workers = [], urls = [];
    const mkWorker = src => {
      try { const u = URL.createObjectURL(new Blob([src], {type: 'text/javascript'})); urls.push(u); const w = new Worker(u); workers.push(w); return w; }
      catch (e) { err('Background workers are unavailable here, so smoothing and the 3D view are off: ' + e.message); return null; }
    };
    let reset = null;
    if (opt.onClose) on($('close'), 'click', () => opt.onClose(api));
    // on a narrow screen the controls are a drawer over the images: never below them
    function setPanel(open) {
      root.dataset.panel = open ? 'open' : '';
      $('ctl').setAttribute('aria-expanded', String(open));
      if (open) { const f = $('rail').querySelector('select, input, button'); if (f) f.focus(); } else $('ctl').focus();
    }
    on($('ctl'), 'click', () => setPanel(root.dataset.panel !== 'open'));
    on($('ctlDone'), 'click', () => setPanel(false));
    if ($('max')) on($('max'), 'click', () => setMax(root.dataset.mode !== 'max'));
    function setMax(m) {
      root.dataset.mode = m ? 'max' : 'inline';
      if ($('max')) { $('max').setAttribute('aria-pressed', String(m)); $('max').textContent = m ? 'Exit full screen' : 'Full screen'; }
      document.documentElement.classList.toggle('ptv-noscroll', m);
      if (opt.onMaximize) opt.onMaximize(m);
      requestAnimationFrame(() => sizeCanvases && sizeCanvases());
    }
    // on the document, before the page's own handlers: Safari doesn't focus a button when it is clicked, so the key may
    // not reach the viewer, and the docs theme marks Escape as handled
    on(document, 'keydown', e => {
      if (e.key !== 'Escape') return;
      if (root.dataset.panel === 'open') { e.preventDefault(); setPanel(false); return; }
      if (root.dataset.mode === 'max') { e.preventDefault(); if (opt.onClose && opt.maximized) opt.onClose(api); else setMax(false); }
    }, true);
    cleanups.push(() => document.documentElement.classList.remove('ptv-noscroll'));
    if (opt.maximized) document.documentElement.classList.add('ptv-noscroll');

    // ---------- the manifest and its layers ----------
    let man, LAY = [];
    const manUrl = typeof opt.manifest === 'string' ? new URL(opt.manifest, location.href).href : null;
    try {
      if (manUrl) {
        const r = await fetch(manUrl, abort ? {signal: abort.signal} : undefined);
        if (!r.ok) throw new Error(`${manUrl} answered ${r.status}`);
        man = await r.json();
      } else man = opt.manifest;
      if (!man || !Array.isArray(man.layers) || !man.layers.length) throw new Error('the manifest lists no images');
      const fileBase = opt.base ? new URL(opt.base, location.href).href : manUrl || location.href;
      $('title').textContent = opt.title || man.title || man.tutorial || 'Images';
      const total = man.layers.reduce((s, l) => s + (l.bytes || 0), 0), got = man.layers.map(() => 0);
      const progress = () => { if (total) setLoading(`Loading the images… ${fmtMB(got.reduce((a, b) => a + b, 0))} of ${fmtMB(total)}`); };
      progress();
      const loaded = await Promise.all(man.layers.map(async (m, i) => {
        const u8 = await fetchBytes(new URL(m.file, fileBase).href, n => { got[i] = n; progress(); }, abort && abort.signal);
        return readNifti(u8, m.file);
      }));
      if (!alive) return api;
      LAY = loaded.map((d, i) => {
        const m = man.layers[i], role = m.role || ROLE[m.kind] || 'base';
        return Object.assign(d, {m, role, base: d.img, ver: 0, set: null});
      });
    } catch (e) {
      if (alive) setLoading('The images did not load: ' + (e && e.message || e));
      return api;
    }
    setLoading('');
    const idx = role => LAY.map((L, i) => L.role === role ? i : -1).filter(i => i >= 0);
    const OVS = idx('overlay'), BASES = idx('base');
    const hasBase = BASES.length > 0;
    function defaults(L) {
      const m = L.m, over = L.role === 'overlay';
      // SPECT and PET start at absolute 0 and the hottest voxel; a CT starts on its named window (phantom, soft...)
      let range = Array.isArray(m.range) && m.range.length === 2 ? m.range.slice() : over ? [0, L.max] : [L.min, L.max];
      const w = m.kind === 'ct' && CT_WINDOWS.find(x => x[0] === m.window);
      if (w) range = [w[2], w[3]];
      return {cmap: LUT[m.colormap] ? m.colormap : over ? 'inferno' : 'gray', lo: range[0], hi: range[1],
        // SPECT and PET over an anatomy image start at 75% opacity (Luke, 9 Oct 2026); on their own, at 100%
        op: m.opacity != null ? m.opacity : over ? (hasBase ? 0.75 : 1) : 1, fwhm: 0, hiAuto: true};
    }
    // images in one scale group (the same units, as the export decides) share their settings, so switching between
    // them keeps the colour scale, and the same colour means the same value
    function assignSettings() {
      const shared = {};
      LAY.forEach(L => {
        const g = L.m.group;
        if (g && shared[g]) L.set = shared[g];
        else { L.set = defaults(L); if (g) shared[g] = L.set; }
      });
    }
    assignSettings();
    const sel = {overlay: OVS.length ? OVS[0] : -1, base: BASES.length ? BASES[0] : -1};
    const facts = [];
    if (man.description) facts.push(man.description);
    const vox = pd => { const p = pd.map(x => +x.toFixed(2)); return (p[0] === p[1] && p[1] === p[2] ? String(p[0]) : p.join(' × ')) + ' mm'; };
    facts.push(LAY.map(L => `${L.m.name} ${vox(L.pd)}`).join(' · '));
    $('facts').textContent = facts.join(' · ');
    if (man.credit) {      // the data's source and licence
      const c = $('credit');
      c.append('Data: ');
      if (/^https:[/][/]/.test(man.credit_url || '')) { const a = document.createElement('a'); a.href = man.credit_url; a.target = '_blank'; a.rel = 'noopener'; a.textContent = man.credit; c.append(a); }
      else c.append(man.credit);
    }
    $('meta').title = $('meta').textContent;   // the line is cut to the width; its whole text on hover

    // ---------- geometry: scanner coordinates (RAS mm) ----------
    const ext = (L, ax) => { const a = L.aff[ax][ax], b = L.aff[ax][3], n = L.dims[ax], e1 = b - 0.5 * a, e2 = b + (n - 0.5) * a; return [Math.min(e1, e2), Math.max(e1, e2)]; };
    let FRAME, CENTER, STEP;
    function setFrame() {
      const F = sel.base >= 0 ? LAY[sel.base] : LAY[sel.overlay];   // the anatomy frames every view, like xlim/ylim in dual_imshow
      FRAME = [0, 1, 2].map(ax => ext(F, ax));
      CENTER = FRAME.map(([a, b]) => (a + b) / 2);
      const shown = [sel.base, sel.overlay].filter(i => i >= 0).map(i => LAY[i]);
      STEP = [0, 1, 2].map(ax => Math.min(...shown.map(L => L.pd[ax])));
    }
    setFrame();
    const startP = () => Array.isArray(man.start_mm) && man.start_mm.length === 3 ? man.start_mm.slice() : CENTER.slice();
    const S = {P: startP(), zoom: 1, pan: [0, 0, 0], view: 'multi', smooth: true};

    function sliceOf(L, vn, k) {
      const V = VIEWS[vn], a = L.aff, d = L.dims, n = V.n, b = L.base;
      const au = V.u[0], av = V.v[0], nu = d[au], nv = d[av], fu = V.u[1] * a[au][au] < 0, fv = V.v[1] * a[av][av] < 0;
      const st = [1, d[0], d[0] * d[1]], data = new Float32Array(nu * nv);
      for (let r = 0; r < nv; r++) {
        const iv = fv ? nv - 1 - r : r;
        for (let c = 0; c < nu; c++) { const iu = fu ? nu - 1 - c : c; data[r * nu + c] = b[iu * st[au] + iv * st[av] + k * st[n]]; }
      }
      const U = sortPair(ext(L, au).map(x => V.u[1] * x)), Vv = sortPair(ext(L, av).map(x => V.v[1] * x));
      return {w: nu, h: nv, data, U, V: Vv};
    }
    function rgbaOf(sl, L) {
      const n = sl.w * sl.h, out = new Uint8ClampedArray(n * 4), d = sl.data, s = L.set, lut = LUT[s.cmap] || LUT.gray;
      const lo = s.lo, span = Math.max(1e-12, s.hi - s.lo), a = Math.round(s.op * 255);
      if (L.role === 'overlay') {
        // The bottom of the colour scale fades in from clear. Otherwise the reconstruction's near-zero voxels around the
        // object would show as a flat sheet of the colormap's darkest colour (hot starts at dark red, not black), the
        // same at every upper limit.
        for (let i = 0; i < n; i++) {
          const v = d[i]; if (!(v > lo)) continue;
          const t = Math.min(1, (v - lo) / span), q = Math.round(t * 255) * 3;
          out[4 * i] = lut[q]; out[4 * i + 1] = lut[q + 1]; out[4 * i + 2] = lut[q + 2]; out[4 * i + 3] = t < FADE ? a * t / FADE : a;
        }
      } else {
        for (let i = 0; i < n; i++) {
          const q = Math.round(clamp((d[i] - lo) / span, 0, 1) * 255) * 3;
          out[4 * i] = lut[q]; out[4 * i + 1] = lut[q + 1]; out[4 * i + 2] = lut[q + 2]; out[4 * i + 3] = a;
        }
      }
      return out;
    }

    // ---------- the three slice views ----------
    const TILE = {};
    root.querySelectorAll('.ptv-tile').forEach(t => { if (t.dataset.v !== 'mip') TILE[t.dataset.v] = {el: t, cv: t.querySelector('canvas'), sl: t.querySelector('input'), cache: new Map()}; });
    function frameOf(vn, W, H) {
      const V = VIEWS[vn], fu = sortPair(FRAME[V.u[0]].map(x => V.u[1] * x)), fv = sortPair(FRAME[V.v[0]].map(x => V.v[1] * x));
      const s = Math.min(W / (fu[1] - fu[0]), H / (fv[1] - fv[0])) * 0.96 * S.zoom;
      return {s, W, H, cU: V.u[1] * (CENTER[V.u[0]] + S.pan[V.u[0]]), cV: V.v[1] * (CENTER[V.v[0]] + S.pan[V.v[0]])};
    }
    function layerImage(T, li, vn, m) {
      const L = LAY[li], V = VIEWS[vn], n = V.n, k = Math.round((S.P[n] - L.aff[n][3]) / L.aff[n][n]);
      if (k < 0 || k >= L.dims[n]) return null;
      const gauss = L.role === 'overlay' && S.smooth;
      // sub-voxel samples only where a voxel covers several screen pixels
      const f = gauss ? clamp(Math.round(m.s * Math.min(L.pd[V.u[0]], L.pd[V.v[0]]) / 3), 1, 4) : 1;
      const key = `${k}|${L.ver}|${gauss}|${f}`;
      let c = T.cache.get(li);
      if (c && c.key === key) return c;
      let sl = sliceOf(L, vn, k);
      if (gauss) sl = gaussUp(sl, f);
      const cv = c ? c.cv : document.createElement('canvas');
      cv.width = sl.w; cv.height = sl.h;
      cv.getContext('2d').putImageData(new ImageData(rgbaOf(sl, L), sl.w, sl.h), 0, 0);
      c = {key, cv, U: sl.U, V: sl.V};
      T.cache.set(li, c);
      return c;
    }
    function drawTile(vn) {
      const T = TILE[vn], cv = T.cv;
      if (!cv.offsetParent) return;
      const W = cv.width, H = cv.height, g = cv.getContext('2d'), V = VIEWS[vn], m = frameOf(vn, W, H), k = dpr();
      g.setTransform(1, 0, 0, 1, 0, 0); g.fillStyle = '#000'; g.fillRect(0, 0, W, H);
      const X = u => W / 2 + (u - m.cU) * m.s, Y = v => H / 2 + (v - m.cV) * m.s;
      for (const li of [sel.base, sel.overlay]) {
        if (li < 0) continue;
        const im = layerImage(T, li, vn, m);
        if (!im) continue;
        g.imageSmoothingEnabled = S.smooth; g.imageSmoothingQuality = 'high';
        g.drawImage(im.cv, X(im.U[0]), Y(im.V[0]), (im.U[1] - im.U[0]) * m.s, (im.V[1] - im.V[0]) * m.s);
      }
      const x = X(V.u[1] * S.P[V.u[0]]), y = Y(V.v[1] * S.P[V.v[0]]);
      g.strokeStyle = 'rgba(77,217,255,0.7)'; g.lineWidth = k; g.beginPath();
      g.moveTo(x, 0); g.lineTo(x, H); g.moveTo(0, y); g.lineTo(W, y); g.stroke();
      g.fillStyle = 'rgba(223,226,234,0.75)'; g.font = `600 ${11 * k}px "IBM Plex Mono",ui-monospace,monospace`; g.textBaseline = 'middle';
      g.textAlign = 'left'; g.fillText(V.o[0], 6 * k, H / 2); g.textAlign = 'right'; g.fillText(V.o[1], W - 6 * k, H / 2);
      g.textAlign = 'center'; g.fillText(V.o[2], W / 2, 10 * k); g.fillText(V.o[3], W / 2, H - 10 * k);
    }
    function syncSlider(vn) {
      const T = TILE[vn], n = VIEWS[vn].n, N = Math.max(1, Math.round((FRAME[n][1] - FRAME[n][0]) / STEP[n]));
      T.sl.max = String(N - 1);
      T.sl.value = String(clamp(Math.floor((S.P[n] - FRAME[n][0]) / STEP[n]), 0, N - 1));
    }
    function valueAt(L, mm) {
      const a = L.aff, d = L.dims, ijk = [0, 1, 2].map(ax => Math.round((mm[ax] - a[ax][3]) / a[ax][ax]));
      if (ijk.some((v, ax) => v < 0 || v >= d[ax])) return NaN;
      return L.base[ijk[0] + d[0] * (ijk[1] + d[1] * ijk[2])];
    }
    const fmtV = (v, L) => !Number.isFinite(v) ? '–' : L.m.kind === 'ct' ? String(Math.round(v)) : fmtN(v);
    function drawStatus() {
      const parts = [];
      for (const li of [sel.overlay, sel.base]) if (li >= 0) {
        const L = LAY[li];
        parts.push(`${L.m.name} ${fmtV(valueAt(L, S.P), L)}${L.m.units ? ' ' + L.m.units : ''}`);
      }
      $('readout').textContent = parts.join(' · ');
      $('where').textContent = S.P.map(v => v.toFixed(1)).join(', ') + ' mm';
      const L = LAY[sel.overlay >= 0 ? sel.overlay : sel.base], lut = LUT[L.set.cmap] || LUT.gray, stops = [];
      for (let i = 0; i <= 16; i++) { const q = Math.min(255, i * 16) * 3; stops.push(`rgb(${lut[q]},${lut[q + 1]},${lut[q + 2]}) ${(i / 16 * 100).toFixed(1)}%`); }
      $('cbGrad').style.background = `linear-gradient(90deg,${stops.join(',')})`;
      $('cbLo').textContent = fmtN(L.set.lo); $('cbHi').textContent = fmtN(L.set.hi) + (L.m.units ? ' ' + L.m.units : '');
    }
    let dirty = false, raf = 0;
    function invalidate() {
      if (dirty || !alive) return;
      dirty = true;
      raf = requestAnimationFrame(() => { dirty = false; if (!alive) return; Object.keys(TILE).forEach(drawTile); Object.keys(TILE).forEach(syncSlider); drawStatus(); });
    }
    cleanups.push(() => cancelAnimationFrame(raf));
    function sizeCanvases() {
      if (!alive) return;
      root.querySelectorAll('.ptv-tile canvas').forEach(c => {
        const r = c.getBoundingClientRect(), w = Math.max(1, Math.round(r.width * dpr())), h = Math.max(1, Math.round(r.height * dpr()));
        if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
      });
      invalidate(); drawMip();
    }
    if (typeof ResizeObserver !== 'undefined') { const ro = new ResizeObserver(sizeCanvases); ro.observe($('tiles')); cleanups.push(() => ro.disconnect()); }
    else on(window, 'resize', sizeCanvases);

    // ---------- pointer: click or drag moves the crosshair, Shift+drag pans, two fingers pinch and pan ----------
    function toWorld(vn, cx, cy) {
      const T = TILE[vn], m = frameOf(vn, T.cv.width, T.cv.height), r = T.cv.getBoundingClientRect(), V = VIEWS[vn];
      const px = (cx - r.left) * dpr(), py = (cy - r.top) * dpr(), U = m.cU + (px - m.W / 2) / m.s, Vv = m.cV + (py - m.H / 2) / m.s;
      return {U, V: Vv, wu: U / V.u[1], wv: Vv / V.v[1], m, px, py};
    }
    const clampP = () => { for (let a = 0; a < 3; a++) S.P[a] = clamp(S.P[a], FRAME[a][0], FRAME[a][1]); };
    // zoom to z, keeping the scanner point under (cx, cy) where it is
    function zoomAt(vn, cx, cy, z, anchor) {
      const V = VIEWS[vn], w = anchor || toWorld(vn, cx, cy);
      S.zoom = clamp(z, 0.5, 16);
      const T = TILE[vn], m = frameOf(vn, T.cv.width, T.cv.height), r = T.cv.getBoundingClientRect();
      const px = (cx - r.left) * dpr(), py = (cy - r.top) * dpr();
      const cU = w.U - (px - m.W / 2) / m.s, cV = w.V - (py - m.H / 2) / m.s;
      S.pan[V.u[0]] = cU / V.u[1] - CENTER[V.u[0]]; S.pan[V.v[0]] = cV / V.v[1] - CENTER[V.v[0]];
      invalidate();
    }
    const stepSlice = (vn, steps) => { const n = VIEWS[vn].n; S.P[n] -= steps * STEP[n]; clampP(); invalidate(); };
    Object.entries(TILE).forEach(([vn, T]) => {
      const V = VIEWS[vn], pts = new Map();
      let mode = null, last = null, pinch = null, wheelAcc = 0, gesture = null;
      const cross = (cx, cy) => { const w = toWorld(vn, cx, cy); S.P[V.u[0]] = w.wu; S.P[V.v[0]] = w.wv; clampP(); invalidate(); };
      const startPinch = () => {
        const [a, b] = [...pts.values()], mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
        pinch = {d0: Math.hypot(a.x - b.x, a.y - b.y) || 1, z0: S.zoom, w: toWorld(vn, mx, my)};
      };
      on(T.cv, 'pointerdown', e => {
        pts.set(e.pointerId, {x: e.clientX, y: e.clientY});
        try { T.cv.setPointerCapture(e.pointerId); } catch (_) {}
        if (pts.size === 2) { mode = 'pinch'; startPinch(); return; }
        if (pts.size > 2) return;
        mode = e.shiftKey || e.button === 1 ? 'pan' : 'cross'; last = [e.clientX, e.clientY];
        if (mode === 'cross') cross(e.clientX, e.clientY);
      });
      on(T.cv, 'pointermove', e => {
        if (!pts.has(e.pointerId)) return;
        pts.set(e.pointerId, {x: e.clientX, y: e.clientY});
        if (mode === 'pinch' && pts.size >= 2) {
          const [a, b] = [...pts.values()];
          zoomAt(vn, (a.x + b.x) / 2, (a.y + b.y) / 2, pinch.z0 * Math.hypot(a.x - b.x, a.y - b.y) / pinch.d0, pinch.w);
        } else if (mode === 'pan') {
          const s = frameOf(vn, T.cv.width, T.cv.height).s / dpr(), dx = (e.clientX - last[0]) / s, dy = (e.clientY - last[1]) / s;
          S.pan[V.u[0]] -= dx / V.u[1]; S.pan[V.v[0]] -= dy / V.v[1]; last = [e.clientX, e.clientY]; invalidate();
        } else if (mode === 'cross') cross(e.clientX, e.clientY);
      });
      const up = e => { pts.delete(e.pointerId); if (pts.size === 0 || mode === 'pinch') mode = pts.size ? 'idle' : null; };
      on(T.cv, 'pointerup', up); on(T.cv, 'pointercancel', up);
      on(T.cv, 'wheel', e => {
        e.preventDefault();
        if (e.ctrlKey) { if (!gesture) zoomAt(vn, e.clientX, e.clientY, S.zoom * Math.pow(1.15, clamp(-e.deltaY / (e.deltaMode === 1 ? 3 : 100), -3, 3))); return; }
        wheelAcc += e.deltaMode === 1 ? e.deltaY * 40 : e.deltaY;
        const steps = Math.trunc(wheelAcc / 100);
        if (!steps) return;
        wheelAcc -= steps * 100;
        stepSlice(vn, Math.sign(steps) * Math.min(Math.abs(steps), 5));
      }, {passive: false});
      // Safari's trackpad pinch arrives as gesture events, not as Ctrl+wheel
      const gxy = e => { const r = T.cv.getBoundingClientRect(); return Number.isFinite(e.clientX) ? [e.clientX, e.clientY] : [r.left + r.width / 2, r.top + r.height / 2]; };
      on(T.cv, 'gesturestart', e => { e.preventDefault(); const [x, y] = gxy(e); gesture = {z0: S.zoom, w: toWorld(vn, x, y), x, y}; });
      on(T.cv, 'gesturechange', e => { e.preventDefault(); if (gesture) zoomAt(vn, gesture.x, gesture.y, gesture.z0 * e.scale, gesture.w); });
      on(T.cv, 'gestureend', e => { e.preventDefault(); gesture = null; });
      on(T.cv, 'keydown', e => {
        const k = e.key;
        if (k === 'ArrowUp' || k === 'PageUp') { e.preventDefault(); stepSlice(vn, k === 'PageUp' ? -5 : -1); }
        else if (k === 'ArrowDown' || k === 'PageDown') { e.preventDefault(); stepSlice(vn, k === 'PageDown' ? 5 : 1); }
        else if (k === '+' || k === '=' || k === '-') {
          e.preventDefault(); const r = T.cv.getBoundingClientRect();
          zoomAt(vn, r.left + r.width / 2, r.top + r.height / 2, S.zoom * (k === '-' ? 1 / 1.25 : 1.25));
        }
      });
      on(T.sl, 'input', () => { const n = V.n; S.P[n] = FRAME[n][0] + (+T.sl.value + 0.5) * STEP[n]; clampP(); invalidate(); });
    });

    // ---------- Smoothing (FWHM in mm) of a layer, in a worker, from the unsmoothed data ----------
    // One job at a time: while the worker is busy, only the newest request waits, so dragging the slider over a
    // large CT doesn't queue a job for every step.
    const gWorker = mkWorker(SMOOTH_SRC), pending = LAY.map(() => 0), asked = LAY.map(() => 0);
    let gBusy = false, gNext = null;
    function smooth(li, fwhm) {
      const L = LAY[li], tag = ++pending[li];
      L.set.fwhm = fwhm; asked[li] = fwhm;
      if (!(fwhm > 0) || !gWorker) {
        L.smoothed = 0;
        if (gNext && gNext.li === li) gNext = null;
        if (L.base !== L.img) { L.base = L.img; L.ver++; followHi(li); invalidate(); mipData(li); }
        return;
      }
      const job = {tag: li * 65536 + tag, li, dims: L.dims, sig: L.pd.map(p => fwhm / 2.3548 / p)};
      if (gBusy) { gNext = job; return; }
      runJob(job);
    }
    function runJob(job) {
      gBusy = true;
      gWorker.postMessage({tag: job.tag, data: LAY[job.li].img.slice(), dims: job.dims, sig: job.sig});
    }
    if (gWorker) gWorker.onmessage = e => {
      const {tag, data} = e.data, li = Math.floor(tag / 65536);
      gBusy = false;
      if (gNext && alive) { const j = gNext; gNext = null; runJob(j); }
      if (!alive || !LAY[li] || tag % 65536 !== pending[li]) return;
      LAY[li].base = data; LAY[li].smoothed = asked[li]; LAY[li].ver++; followHi(li); invalidate(); mipData(li);
    };
    // Smoothing lowers peaks. While the reader hasn't set the upper limit, it follows: scaled by how much the hottest
    // spot (the starting point, or this image's hottest voxel) drops, so smoothed spheres keep their brightness and
    // visibly spread, instead of only dimming.
    function hotMm(L) {
      if (!L.hot) { let k = 0; for (let i = 1; i < L.img.length; i++) if (L.img[i] > L.img[k]) k = i;
        const d = L.dims, ijk = [k % d[0], Math.floor(k / d[0]) % d[1], Math.floor(k / (d[0] * d[1]))];
        L.hot = [0, 1, 2].map(a => L.aff[a][a] * ijk[a] + L.aff[a][3]); }
      return L.hot;
    }
    function meanAt(arr, L, mm) {   // the mean over the 3 x 3 x 3 voxels around a point
      const a = L.aff, d = L.dims, c = [0, 1, 2].map(ax => Math.round((mm[ax] - a[ax][3]) / a[ax][ax]));
      let s = 0, n = 0;
      for (let k = c[2] - 1; k <= c[2] + 1; k++) for (let j = c[1] - 1; j <= c[1] + 1; j++) for (let i = c[0] - 1; i <= c[0] + 1; i++) {
        if (i < 0 || j < 0 || k < 0 || i >= d[0] || j >= d[1] || k >= d[2]) continue;
        s += arr[i + d[0] * (j + d[1] * k)]; n++;
      }
      return n ? s / n : NaN;
    }
    function followHi(li) {
      const L = LAY[li];
      if (L.role !== 'overlay' || !L.set.hiAuto) return;
      const start = Array.isArray(man.start_mm) && Number.isFinite(meanAt(L.img, L, man.start_mm)) ? man.start_mm : hotMm(L);
      const before = meanAt(L.img, L, start), after = meanAt(L.base, L, start);
      const r = L.base === L.img || !(before > 0) ? 1 : after / before;
      L.set.hi = defaults(L).hi * r;
      LAY.forEach(M => { if (M.set === L.set) M.ver++; });
      const c = cardO;
      if (c && sel.overlay === li) c.show();
      drawStatus(); requestMip();
    }

    // ---------- 3D view ----------
    const mWorker = mkWorker(MIP_SRC), mipCv = $('mip'), mipOff = document.createElement('canvas');
    let mipTheta = 0, mipRot = !reduceMotion, mipDrag = false, mipBusy = false, mipAgain = false, mipFrame = false, mipZoom = 1, mipSeen = true;
    const mipShown = () => !!mipCv.offsetParent && mipSeen;
    function mipGrid() {
      const ext3 = FRAME.map(([a, b]) => b - a), shown = [sel.base, sel.overlay].filter(i => i >= 0).map(i => LAY[i]);
      const h = Math.max(2, ...shown.map(L => Math.min(...L.pd)).concat([Math.max(...ext3) / 128]));
      return {h, x0: FRAME[0][0] + h / 2, y0: FRAME[1][0] + h / 2, z0: FRAME[2][0] + h / 2,
        gx: Math.max(1, Math.floor(ext3[0] / h)), gy: Math.max(1, Math.floor(ext3[1] / h)), gz: Math.max(1, Math.floor(ext3[2] / h))};
    }
    function mipAll() {
      if (!mWorker) return;
      mWorker.postMessage({cmd: 'grid', grid: mipGrid()});
      // both images first, then one frame, so the first frame already has its colour image
      if (sel.base >= 0) mipData(sel.base, true); else mWorker.postMessage({cmd: 'nobase'});
      if (sel.overlay >= 0) mipData(sel.overlay, true); else mWorker.postMessage({cmd: 'noov'});
      requestMip();
    }
    function mipData(li, quiet) {
      if (!mWorker || (li !== sel.base && li !== sel.overlay)) return;
      const L = LAY[li];
      if (L.role === 'overlay') mWorker.postMessage({cmd: 'ov', img: L.base, dims: L.dims, aff: L.aff});
      else mWorker.postMessage({cmd: 'base', img: L.base, dims: L.dims, aff: L.aff, kind: L.m.kind, scale: L.m.kind === 'ct' ? 1 : Math.max(1e-12, L.max)});
      if (!quiet) requestMip();
    }
    function requestMip() {
      if (!mWorker || !mipShown() || !alive) return;
      if (mipBusy) { mipAgain = true; return; }
      mipBusy = true;
      const O = sel.overlay >= 0 ? LAY[sel.overlay].set : null, B = sel.base >= 0 ? LAY[sel.base].set : null;
      mWorker.postMessage({cmd: 'render', p: {th: mipTheta * Math.PI / 180, lut: LUT[O ? O.cmap : 'gray'] || LUT.gray, lo: O ? O.lo : 0, hi: O ? O.hi : 1,
        op: O ? (B ? Math.min(1, 0.4 + O.op) : 1) : 0, ctw: B ? B.op : 0, smooth: S.smooth}});
    }
    if (mWorker) mWorker.onmessage = e => {
      const d = e.data;
      mipBusy = false;
      if (!alive) return;
      if (d.W) { mipOff.width = d.W; mipOff.height = d.H; mipOff.getContext('2d').putImageData(new ImageData(d.rgba, d.W, d.H), 0, 0); mipFrame = true; drawMip(); }
      if (mipAgain) { mipAgain = false; requestMip(); }
    };
    function drawMip() {
      if (!mipFrame || !mipCv.offsetParent) return;
      const g = mipCv.getContext('2d'), W = mipCv.width, H = mipCv.height;
      g.fillStyle = '#000'; g.fillRect(0, 0, W, H);
      const s = Math.min(W / mipOff.width, H / mipOff.height) * 0.92 * mipZoom, w = mipOff.width * s, h = mipOff.height * s;
      g.imageSmoothingEnabled = S.smooth; g.imageSmoothingQuality = 'high';
      g.drawImage(mipOff, (W - w) / 2, (H - h) / 2, w, h);
      $('miplab').textContent = `${sel.overlay >= 0 ? 'MIP' : 'Projection'} · ${Math.round(((mipTheta % 360) + 360) % 360)}°`;
    }
    let lastT = 0, tickId = 0;
    const tick = t => {
      if (!alive) return;
      if (mipRot && !mipDrag && mipShown() && !document.hidden) { const dt = lastT ? Math.min(0.1, (t - lastT) / 1000) : 0; mipTheta = (mipTheta + dt * 30) % 360; requestMip(); }
      lastT = t || 0; tickId = requestAnimationFrame(tick);
    };
    tickId = requestAnimationFrame(tick);
    cleanups.push(() => cancelAnimationFrame(tickId));
    if (typeof IntersectionObserver !== 'undefined') {
      const io = new IntersectionObserver(es => { mipSeen = es[es.length - 1].isIntersecting; if (mipSeen) requestMip(); });
      io.observe(mipCv); cleanups.push(() => io.disconnect());
    }
    (function () {
      const pts = new Map();
      let x0 = null, pinch = null;
      on(mipCv, 'pointerdown', e => {
        pts.set(e.pointerId, {x: e.clientX, y: e.clientY});
        try { mipCv.setPointerCapture(e.pointerId); } catch (_) {}
        if (pts.size === 2) { const [a, b] = [...pts.values()]; pinch = {d0: Math.hypot(a.x - b.x, a.y - b.y) || 1, z0: mipZoom}; x0 = null; return; }
        x0 = e.clientX; mipDrag = true;
      });
      on(mipCv, 'pointermove', e => {
        if (!pts.has(e.pointerId)) return;
        pts.set(e.pointerId, {x: e.clientX, y: e.clientY});
        if (pinch && pts.size >= 2) { const [a, b] = [...pts.values()]; mipZoom = clamp(pinch.z0 * Math.hypot(a.x - b.x, a.y - b.y) / pinch.d0, 0.5, 8); drawMip(); return; }
        if (x0 === null) return;
        mipTheta += (e.clientX - x0) * 0.6; x0 = e.clientX; requestMip();
      });
      const up = e => { pts.delete(e.pointerId); if (pts.size < 2) pinch = null; if (!pts.size) { x0 = null; mipDrag = false; } };
      on(mipCv, 'pointerup', up); on(mipCv, 'pointercancel', up);
      on(mipCv, 'wheel', e => { if (!e.ctrlKey) return; e.preventDefault(); mipZoom = clamp(mipZoom * (e.deltaY < 0 ? 1.15 : 1 / 1.15), 0.5, 8); drawMip(); }, {passive: false});
      let gz = 1;
      on(mipCv, 'gesturestart', e => { e.preventDefault(); gz = mipZoom; });
      on(mipCv, 'gesturechange', e => { e.preventDefault(); mipZoom = clamp(gz * e.scale, 0.5, 8); drawMip(); });
      on(mipCv, 'keydown', e => {
        if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') { e.preventDefault(); mipTheta += e.key === 'ArrowLeft' ? -10 : 10; requestMip(); }
        else if (e.key === ' ') { e.preventDefault(); setRot(!mipRot); }
      });
    })();
    function setRot(v) { mipRot = v; $('rot').setAttribute('aria-pressed', String(v)); $('rot').textContent = v ? 'Rotating' : 'Paused'; }
    on($('rot'), 'click', () => setRot(!mipRot));

    // ---------- layer controls ----------
    const cards = $('cards');
    // the header's Image switch serves the colour images, or the grey ones when only they come in several
    const headRole = OVS.length > 1 ? 'overlay' : BASES.length > 1 ? 'base' : null;
    function card(role) {
      const list = role === 'overlay' ? OVS : BASES;
      if (!list.length) return null;
      const r = role === 'overlay' ? 'o' : 'b', I = s => id(r + s), first = LAY[list[0]];
      const pick = list.length > 1 && role !== headRole
        ? `<div class="ptv-ctl"><label for="${I('Pick')}">Image</label><select id="${I('Pick')}">${list.map(i => `<option value="${i}">${esc(LAY[i].m.label || LAY[i].m.name)}</option>`).join('')}</select></div>` : '';
      const isCT = l => l.m.kind === 'ct';
      const win = role === 'base'
        ? `<div class="ptv-ctl"><label for="${I('Win')}">Window</label><select id="${I('Win')}"></select></div>` : '';
      const el = document.createElement('div');
      el.className = 'ptv-layer';
      el.innerHTML = `<h3><span id="${I('Name')}">${esc(first.m.name)}</span><small id="${I('Units')}"></small></h3>${pick}` +
        `<div class="ptv-ctl"><label for="${I('Cmap')}">Colormap</label><select id="${I('Cmap')}">${CMAP_ORDER.map(c => `<option value="${c}">${c}</option>`).join('')}</select></div>${win}` +
        `<div class="ptv-ctl"><label for="${I('LoR')}">Lower limit</label><input type="range" id="${I('LoR')}" aria-label="Lower limit slider"><input type="number" id="${I('Lo')}" step="any" inputmode="decimal" aria-label="Lower limit"></div>` +
        `<div class="ptv-ctl"><label for="${I('HiR')}">Upper limit</label><input type="range" id="${I('HiR')}" aria-label="Upper limit slider"><input type="number" id="${I('Hi')}" step="any" inputmode="decimal" aria-label="Upper limit"></div>` +
        `<div class="ptv-ctl"><label for="${I('Fw')}">Smoothing</label><input type="range" id="${I('Fw')}" min="0" max="${role === 'overlay' ? 20 : 10}" step="1" value="0"><output id="${I('FwV')}" for="${I('Fw')}">0 mm</output></div>` +
        `<div class="ptv-ctl"><label for="${I('Op')}">Opacity</label><input type="range" id="${I('Op')}" min="0" max="100" step="5"><output id="${I('OpV')}" for="${I('Op')}"></output></div>`;
      cards.appendChild(el);
      const E = s => document.getElementById(I(s)), cur = () => LAY[sel[role]];
      const windows = L => { if (isCT(L)) return CT_WINDOWS; const d = defaults(L); return [['default', 'Default', d.lo, d.hi], ['full', 'Full range', L.min, L.max]]; };
      function show() {    // put the selected layer's settings into the controls
        const L = cur(), s = L.set, over = role === 'overlay';
        const lo = Math.min(L.min, s.lo, over ? 0 : isCT(L) ? -1350 : L.min), hi = Math.max(over ? L.max * 1.5 : L.max, s.hi, isCT(L) ? 1050 : -Infinity);
        const step = (hi - lo) / 1000 || 1;
        E('Name').textContent = L.m.name; E('Units').textContent = L.m.units || '';
        E('Cmap').value = s.cmap;
        ['LoR', 'HiR'].forEach(k => { const x = E(k); x.min = String(lo); x.max = String(hi); x.step = String(step); });
        E('Lo').value = String(+s.lo.toPrecision(6)); E('Hi').value = String(+s.hi.toPrecision(6)); E('LoR').value = String(s.lo); E('HiR').value = String(s.hi);
        E('Fw').value = String(s.fwhm); E('FwV').textContent = s.fwhm + ' mm';
        E('Op').value = String(Math.round(s.op * 100)); E('OpV').textContent = Math.round(s.op * 100) + '%';
        if (E('Win')) {
          const ws = windows(L);
          E('Win').innerHTML = ws.map(([k, t]) => `<option value="${k}">${esc(t)}</option>`).join('') + '<option value="custom" hidden>Custom</option>';
          const hit = ws.find(w => w[2] === s.lo && w[3] === s.hi);
          E('Win').value = hit ? hit[0] : 'custom';
        }
        if (E('Pick')) E('Pick').value = String(sel[role]);
      }
      function changed(L) { LAY.forEach(M => { if (M.set === L.set) M.ver++; }); invalidate(); requestMip(); }
      function limits(lo, hi) {
        const L = cur(), ok = Number.isFinite(lo) && Number.isFinite(hi) && lo < hi;
        ['Lo', 'Hi'].forEach(k => E(k).setAttribute('aria-invalid', String(!ok)));
        if (!ok) { err('The lower limit must be below the upper limit.'); return; }
        err(''); L.set.lo = lo; L.set.hi = hi; L.set.hiAuto = false; changed(L);
        if (E('Win')) { const hit = windows(L).find(w => w[2] === lo && w[3] === hi); E('Win').value = hit ? hit[0] : 'custom'; }
      }
      on(E('LoR'), 'input', () => { E('Lo').value = String(+(+E('LoR').value).toPrecision(6)); limits(+E('LoR').value, parseFloat(E('Hi').value)); });
      on(E('HiR'), 'input', () => { E('Hi').value = String(+(+E('HiR').value).toPrecision(6)); limits(parseFloat(E('Lo').value), +E('HiR').value); });
      ['Lo', 'Hi'].forEach(k => on(E(k), 'change', () => {
        const lo = parseFloat(E('Lo').value), hi = parseFloat(E('Hi').value);
        if (Number.isFinite(lo)) E('LoR').value = String(lo);
        if (Number.isFinite(hi)) E('HiR').value = String(hi);
        limits(lo, hi);
      }));
      on(E('Cmap'), 'change', () => { const L = cur(); L.set.cmap = E('Cmap').value; changed(L); drawStatus(); });
      on(E('Op'), 'input', () => { const L = cur(), v = +E('Op').value; E('OpV').textContent = v + '%'; L.set.op = v / 100; changed(L); });
      on(E('Fw'), 'input', () => { const v = +E('Fw').value; E('FwV').textContent = v + ' mm'; smooth(sel[role], v); });
      if (E('Win')) on(E('Win'), 'change', () => {
        const w = windows(cur()).find(x => x[0] === E('Win').value);
        if (!w) return;
        E('Lo').value = String(w[2]); E('Hi').value = String(w[3]); E('LoR').value = String(w[2]); E('HiR').value = String(w[3]);
        limits(w[2], w[3]);
      });
      if (E('Pick')) on(E('Pick'), 'change', () => pickLayer(role, +E('Pick').value));
      return {show};
    }
    const cardO = card('overlay'), cardB = card('base');
    function pickLayer(role, li) {
      sel[role] = li; setFrame(); clampP();
      Object.values(TILE).forEach(T => T.cache.clear());
      const L = LAY[li];
      if ((L.smoothed || 0) !== (L.set.fwhm || 0)) smooth(li, L.set.fwhm);   // a shared smoothing width
      const c = role === 'overlay' ? cardO : cardB;
      if (c) c.show();
      if (headRole === role) $('pick').value = String(li);
      invalidate(); mipAll();
    }
    if (headRole) {
      const list = headRole === 'overlay' ? OVS : BASES;
      $('pick').innerHTML = list.map(i => `<option value="${i}">${esc(LAY[i].m.label || LAY[i].m.name)}</option>`).join('');
      $('pickWrap').hidden = false;
      on($('pick'), 'change', () => pickLayer(headRole, +$('pick').value));
    }

    on($('interp'), 'change', () => { S.smooth = $('interp').checked; invalidate(); requestMip(); drawMip(); });
    on($('views'), 'click', e => {
      const b = e.target.closest('button[data-view]');
      if (!b) return;
      setView(b.dataset.view);
    });
    function setView(v) {
      S.view = v; $('tiles').dataset.view = v;
      root.querySelectorAll(`#${id('views')} button`).forEach(x => x.setAttribute('aria-pressed', String(x.dataset.view === v)));
      requestAnimationFrame(() => { sizeCanvases(); requestMip(); });
    }

    reset = function () {
      LAY.forEach((L, i) => { L.base = L.img; L.smoothed = 0; L.ver++; pending[i]++; });
      assignSettings();
      sel.overlay = OVS.length ? OVS[0] : -1; sel.base = BASES.length ? BASES[0] : -1;
      if (headRole) $('pick').value = String(sel[headRole]);
      setFrame();
      S.P = startP(); S.zoom = 1; S.pan = [0, 0, 0]; S.smooth = true; $('interp').checked = true;
      Object.values(TILE).forEach(T => T.cache.clear());
      if (cardO) cardO.show();
      if (cardB) cardB.show();
      err('');
      mipZoom = 1; mipTheta = 0; setRot(!reduceMotion);
      setView(opt.view || (root.clientWidth < 520 && coarse ? 'axial' : 'multi'));
      mipAll();
    };
    on($('reset'), 'click', () => reset());
    reset();
    return api;
  }

  window.PTViewer = {mount, version: '1.0'};
})();
