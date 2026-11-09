import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { transform } from 'esbuild';
const {code}=await transform(readFileSync('src/components/graphPreviewData.ts','utf8'),{loader:'ts',format:'esm'});
const {normalisePreviewGraph}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
test('API graph preserves linked nodes without executing content',()=>{
 const data={'1':{class_type:'Loader',inputs:{name:'<script>unsafe</script>'}},'2':{class_type:'Sampler',inputs:{model:['1',0]}}};
 const before=JSON.stringify(data);const graph=normalisePreviewGraph(data);
 assert.equal(graph.nodes.length,2);assert.equal(graph.edges.length,1);
 assert.equal(graph.edges[0].source,'1');assert.equal(graph.edges[0].target,'2');
 assert.ok(graph.nodes.every(n=>Number.isFinite(n.x)&&Number.isFinite(n.y)));
 assert.equal(JSON.stringify(data),before);
});
test('cyclic graph has finite layout',()=>{
 const graph=normalisePreviewGraph({'1':{class_type:'A',inputs:{x:['2',0]}},'2':{class_type:'B',inputs:{x:['1',0]}}});
 assert.equal(graph.nodes.length,2);assert.ok(Number.isFinite(graph.width));assert.ok(Number.isFinite(graph.height));
});
