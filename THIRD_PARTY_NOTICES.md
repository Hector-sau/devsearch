# Third-party attribution and terms / 第三方来源与条款

DevSearch uses pinned upstream documentation and a pretrained encoder in its experiments. The repository contains implementation, miniature original fixtures, compact evidence and demonstration images. It does not distribute the private corpus, query/review records, pretrained model weights or training checkpoints. These notices identify the upstream terms; they do not choose a license for DevSearch's original code.

DevSearch 的实验使用固定版本的上游文档及预训练编码器。本仓库保存实现、原创微型样例、小型证据和演示图片，不分发私有语料、问题／审查记录、预训练权重或训练断点。以下说明标明上游条款，不为 DevSearch 原创代码选择许可证。

## PyTorch Tutorials / PyTorch 教程

- Source: [`pytorch/tutorials`](https://github.com/pytorch/tutorials/tree/11512db7cbbcc4b530ecfc63205d3fed13fa190d), revision `11512db7cbbcc4b530ecfc63205d3fed13fa190d`.
- Terms: [the exact snapshot LICENSE](https://github.com/pytorch/tutorials/blob/11512db7cbbcc4b530ecfc63205d3fed13fa190d/LICENSE), **BSD 3-Clause**.

来源与条款见上述固定提交及 LICENSE，为 **BSD 三条款许可证**。

Demonstration screenshots can display short passages from this source. The original copyright notice, conditions and disclaimer are reproduced verbatim below. Retain them with distributed source text or derived documentation/binary material; no upstream endorsement is claimed. A future distribution of the restored corpus must also preserve applicable notices, including any additional terms attached to individual content.

演示截图可能显示来源中的简短段落。下方逐字保留原版权说明、条件和免责声明；分发来源文本、衍生文档或二进制材料时须保留相应说明，不声称上游为项目背书。未来若分发恢复后的语料，还须保留适用的其他内容条款。

```text
BSD 3-Clause License

Copyright (c) 2017-2022, Pytorch contributors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## MiniLM / MiniLM 编码器

- Model: [`microsoft/MiniLM-L12-H384-uncased`](https://huggingface.co/microsoft/MiniLM-L12-H384-uncased/tree/44acabbec0ef496f6dbc93adadea57f376b7c0ec), revision `44acabbec0ef496f6dbc93adadea57f376b7c0ec`.
- The [README at that exact revision](https://huggingface.co/microsoft/MiniLM-L12-H384-uncased/blob/44acabbec0ef496f6dbc93adadea57f376b7c0ec/README.md) declares **MIT**. Its [revision metadata](https://huggingface.co/api/models/microsoft/MiniLM-L12-H384-uncased/revision/44acabbec0ef496f6dbc93adadea57f376b7c0ec) confirms the SHA and license declaration; that model snapshot does not include a separate LICENSE file.
- The model README points to [Microsoft UniLM/MiniLM](https://github.com/microsoft/unilm/tree/master/minilm). The [current repository LICENSE](https://github.com/microsoft/unilm/blob/master/LICENSE) is MIT, copyright Microsoft Corporation. This current supplemental link is not a claim that a separate UniLM commit was pinned by our model manifest.
- Research reference: Wenhui Wang et al. (2020), [MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers](https://arxiv.org/abs/2002.10957).

上述固定模型 README 明确声明 **MIT**，固定版本元数据核对了 SHA 和许可证说明，该模型快照没有单独 LICENSE 文件。README 链接的 Microsoft UniLM 当前仓库 LICENSE 也是 MIT，版权归 Microsoft Corporation；该补充链接不表示本项目另行固定了 UniLM 提交。研究引用见上述论文。

Weights are not a Git or version-1 wheel attachment. If weights or substantial upstream software/model material are later redistributed, retain their applicable copyright and permission notices with the distribution. A converted serialization keeps the upstream attribution but does not replace this project's original asset-byte checks.

权重不作为 Git 或第一版 wheel 附件。后续若分发权重或大量上游软件／模型内容，须在交付中保留适用版权及许可说明。序列化转换仍须保留上游来源，也不能替代本项目对原资产字节的校验。

## DevSearch and dependencies / 本项目与依赖

DevSearch's original code and documentation are licensed under **Apache-2.0**, as authorized for the public release on 2026-10-10. See [LICENSE](LICENSE) and [NOTICE](NOTICE). This applies to original project material; it does not replace the upstream licenses above. Python dependencies remain separately licensed upstream packages; the platform lock records versions and hashes and does not relicense them. The current project wheel does not vendor those dependencies.

所有者于2026-10-10授权公开发布，DevSearch 原创代码与文档采用 **Apache-2.0**，见 LICENSE 和 NOTICE；不替代上述上游许可证。Python 依赖遵循各上游许可证；平台锁记录版本与哈希，不重新授权依赖。当前项目 wheel 不内嵌这些依赖。

Attribution sources verified on 2026-10-09 using the exact tutorial license and pinned model README/metadata; no model weights were downloaded for this check.

上述来源于 2026-10-09 通过准确教程许可证和固定模型 README／元数据核验；本次未下载模型权重。
