#!/usr/bin/env python3
"""Complete AEROTICA demonstration."""

import numpy as np
from aerotica.ake import AKEComposite
from aerotica.parameters import KED, TII, VSR, THD


def main():
    """Run complete demo."""
    print("="*60)
    print("AEROTICA Complete Demonstration")
    print("="*60)
    
    # 1. Parameter computation
    print("\n1️⃣ Computing individual parameters...")
    
    ked = KED()
    ked_value = ked.compute(10.0)
    print(f"   KED: {ked_value:.1f} W/m²")
    
    tii = TII()
    tii_value = tii.compute([10, 12, 11, 13, 12], method='scalar')
    print(f"   TII: {tii_value:.3f}")
    
    vsr = VSR()
    vsr_result = vsr.compute(10.0, 100)
    print(f"   VSR: {vsr_result['vsr']:.3f} (α={vsr_result['alpha']:.2f})")
    
    thd = THD()
    thd_value = 0.75
    print(f"   THD: {thd_value:.2f}")
    print(f"   Gust probability: {thd.gust_probability(thd_value):.1%}")
    
    # 2. AKE composite
    print("\n2️⃣ Computing AKE composite index...")
    
    ake = AKEComposite(
        site_id="demo_site",
        climate_zone="temperate",
        site_type="demo"
    )
    
    ake.load_parameters({
        "KED": 0.83, "TII": 0.76, "VSR": 0.89,
        "AOD": 0.34, "THD": thd_value, "PGF": 0.65,
        "HCI": 0.59, "ASI": 0.71, "LRC": 0.44
    })
    
    result = ake.compute()
    
    print(f"   AKE Score: {result['score']:.3f}")
    print(f"   Classification: {result['classification']}")
    print(f"   Gust Risk: {result['gust_risk']}")
    print(f"   Confidence: {result['confidence']:.1%}")
    
    # 3. Contributions
    print("\n3️⃣ Parameter contributions:")
    for param, contrib in sorted(
        result['contributions'].items(),
        key=lambda x: -x[1]['contribution']
    )[:5]:
        print(f"   {param}: {contrib['contribution']:.3f} "
              f"(score={contrib['score']:.2f}, weight={contrib['weight']:.2f})")
    
    print("\n✅ Demo completed successfully!")


if __name__ == "__main__":
    main()
