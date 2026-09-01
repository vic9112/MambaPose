from torch import nn


def test_parameter_inventory_separates_trainable_and_total():
    from mambapose_opt.inventory import count_parameters

    model = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2, bias=False))
    model[1].weight.requires_grad_(False)

    inv = count_parameters(model)

    assert inv.total == 21
    assert inv.trainable == 15
    assert inv.bytes_by_dtype['torch.float32'] == 84
    assert inv.by_prefix == {'0': 15, '1': 6}


class PoseInteraction(nn.Module):
    pass


class CrossScan(nn.Module):
    pass


class SelectiveScan(nn.Module):
    pass


class CrossMerge(nn.Module):
    pass


def test_module_inventory_records_dynamic_and_custom_scan_hazards():
    from mambapose_opt.inventory import collect_module_inventory

    model = nn.Module()
    model.pif = PoseInteraction()
    model.cross_scan = CrossScan()
    model.selective_scan = SelectiveScan()
    model.cross_merge = CrossMerge()

    records = {record.name: record for record in collect_module_inventory(model)}

    assert records['pif'].kind == 'PoseInteraction'
    assert records['pif'].hazard == 'dynamic top-k'
    assert records['cross_scan'].hazard == 'custom VMamba scan'
    assert records['selective_scan'].hazard == 'custom VMamba scan'
    assert records['cross_merge'].hazard == 'custom VMamba scan'
