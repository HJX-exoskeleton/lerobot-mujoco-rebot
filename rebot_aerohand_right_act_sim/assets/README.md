# 任务场景位置说明

`rebot_act_sim` 的任务场景 XML 存放在本包的 `assets/` 目录中。本包的
arm + hand 场景无法照搬这一布局：`rebor_arm_6dof.xml` 内部以相对路径嵌套
include `aerohand_right_body.xml`，且手部网格与纹理均以相对路径引用；
MuJoCo 仅当顶层 XML 与这些文件位于同一目录时才能正确解析全部嵌套路径。

因此任务场景统一放在资产包的场景目录中：

```text
asset_rebot_aerohand_right/mujoco_xml/rebotarm_aerohand_act_cylinder.xml
```

配置 `configs/aerohand_act_sim.yaml` 的 `environment.xml` 指向该文件。
若需要派生新任务场景，请在同一目录新建文件并保持 include 与上面三个
组件（`rebotarm_aerohand_scene.xml`、`rebor_arm_6dof.xml`、
`aerohand_right.xml`）同目录引用的写法，不要复制资产文件。
