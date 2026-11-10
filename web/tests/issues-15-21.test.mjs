import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';

const bundled = await build({
  stdin: {
    contents: `
      export { activationSmokeParams, missingActivationInputs } from './src/lib/activation-resume.ts';
      export { selectActivationProjectId, projectOptionLabel } from './src/lib/project-label.ts';
      export { resolveInstanceSubmission } from './src/lib/instance-submission.ts';
      export { mergeWorkflowDraft } from './src/lib/workflow-draft.ts';
      export { workflowTitle, workflowUsage } from './src/lib/workflow-copy.ts';
      export { formatDuration, timingRows } from './src/lib/job-history.ts';
      export { taskStatusNote } from './src/lib/job-detail-copy.ts';
    `,
    resolveDir: fileURLToPath(new URL('..', import.meta.url)),
    sourcefile: 'issues-15-21-entry.ts',
    loader: 'ts',
  },
  bundle: true,
  format: 'esm',
  write: false,
  platform: 'node',
  logLevel: 'silent',
});
const {
  activationSmokeParams,
  missingActivationInputs,
  selectActivationProjectId,
  projectOptionLabel,
  resolveInstanceSubmission,
  mergeWorkflowDraft,
  workflowTitle,
  workflowUsage,
  formatDuration,
  timingRows,
  taskStatusNote,
} = await import('data:text/javascript;base64,' + Buffer.from(bundled.outputFiles[0].text).toString('base64'));

const imageWorkflow = {
  parameters: {
    first_frame: { type: 'image', label: '首帧图', required: true },
    last_frame: { type: 'image', label: '尾帧图', required: false },
    prompt: { type: 'string', label: '提示词', default: 'hello' },
    width: { type: 'integer', default: 480, min: 256, max: 1088, step: 32 },
    height: { type: 'integer', default: 864, min: 256, max: 1088, step: 32 },
  },
};

const imageInputs = [
  { key: 'first_frame', label: '首帧图', type: 'image', required: true },
  { key: 'last_frame', label: '尾帧图', type: 'image', required: false },
];

test('activation smoke keeps required images and does not invent empty ones', () => {
  assert.deepEqual(missingActivationInputs(imageInputs, {}), ['首帧图']);
  assert.deepEqual(missingActivationInputs(imageInputs, { first_frame: 'asset-1' }), []);
  assert.deepEqual(activationSmokeParams(imageInputs, { first_frame: 'asset-1', last_frame: '' }), {
    params: { first_frame: 'asset-1' },
    assetIds: ['asset-1'],
  });
});

test('activation uses the selected project instead of the first or a mock project', () => {
  const projects = [
    { id: 'mock-a', title: '回归验收（MOCK）' },
    { id: 'colab', title: 'colab' },
    { id: 'mock-b', title: '回归验收（MOCK）' },
  ];
  assert.equal(selectActivationProjectId(projects, 'colab'), 'colab');
  assert.equal(selectActivationProjectId(projects, ''), '');
  assert.equal(selectActivationProjectId(projects, 'missing'), '');
  assert.equal(projectOptionLabel(projects[0], projects), '回归验收（MOCK） · mock-a');
  assert.equal(projectOptionLabel(projects[1], projects), 'colab');
});

test('a released machine blocks submission and keeps the selected account', () => {
  const instances = [
    { id: 'dead', state: 'released', account_id: 'acct-user', instance_label: '旧机器' },
    { id: 'live', state: 'ready', account_id: 'acct-other', instance_label: '另一台' },
  ];
  const blocked = resolveInstanceSubmission({ accountId: 'acct-user', instanceId: 'dead' }, instances);
  assert.equal(blocked.ok, false);
  assert.equal(blocked.instanceId, null);
  assert.equal(blocked.accountId, 'acct-user');
  assert.match(blocked.message, /没有提交/);
  assert.match(blocked.message, /没有换账号/);

  const automatic = resolveInstanceSubmission({ accountId: 'acct-user', instanceId: null }, instances);
  assert.equal(automatic.ok, true);
  assert.equal(automatic.allocation, 'auto');
  assert.equal(automatic.accountId, 'acct-user');
  assert.equal(automatic.instanceId, null);

  const reusable = resolveInstanceSubmission({ accountId: 'acct-other', instanceId: 'live' }, instances);
  assert.equal(reusable.ok, true);
  assert.equal(reusable.allocation, 'existing');
  assert.equal(reusable.instanceId, 'live');
});

test('workflow switch keeps each draft and does not wipe a shared prompt for an incompatible size', () => {
  const krea = {
    parameters: {
      prompt: { type: 'string', default: '默认提示' },
      width: { type: 'integer', default: 832, min: 256, max: 2048, step: 16 },
      height: { type: 'integer', default: 1248, min: 256, max: 2048, step: 16 },
    },
  };
  const current = { prompt: 'a quiet garden', width: 832, height: 1248 };
  const firstVisit = mergeWorkflowDraft(imageWorkflow, null, { prompt: current.prompt, width: current.width, height: current.height });
  assert.equal(firstVisit.prompt, 'a quiet garden');
  assert.equal(firstVisit.width, imageWorkflow.parameters.width.default);
  assert.equal(firstVisit.height, imageWorkflow.parameters.height.default);

  const restored = mergeWorkflowDraft(krea, current, { prompt: '别的工作流提示', width: 480, height: 864 });
  assert.equal(restored.prompt, 'a quiet garden');
  assert.equal(restored.width, 832);
  assert.equal(restored.height, 1248);
});

test('workflow names stay readable while the stable key remains available', () => {
  const workflow = { workflow_key: 'minimax-h3-i2v', manifest: { title: 'MiniMax H3 图生视频（带声音）', description: '上传首帧。' } };
  assert.equal(workflowTitle(workflow), 'MiniMax H3 图生视频（带声音）');
  assert.match(workflowUsage(workflow), /首帧/);
  assert.equal(workflowTitle({ workflow_key: 'demo-mock' }), '离线演示');
  assert.match(workflowUsage({ workflow_key: 'demo-mock' }), /不扣费/);
  assert.equal(workflow.workflow_key, 'minimax-h3-i2v');
});

test('stage timings keep fractional seconds and separate a finished task from publication', () => {
  assert.equal(formatDuration(122.4), '2分2.4秒');
  assert.equal(formatDuration(29.1), '29.1秒');
  assert.equal(formatDuration(40), '40秒');
  const rows = timingRows({ timings: { sampling: 224.4, restore_weights: 122.4, comfy_start: 29.1 } });
  assert.deepEqual(rows.map((row) => row.key), ['restore_weights', 'comfy_start', 'sampling']);
  assert.equal(rows[0].value, '2分2.4秒');
  assert.match(taskStatusNote({ state: 'completed', workflow_smoke: false }), /不表示工作流已经发布/);
  assert.match(taskStatusNote({ state: 'completed', workflow_smoke: true }), /冒烟/);
});
