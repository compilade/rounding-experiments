{
  outputs = {
    self,
    nixpkgs,
  }: let
    lib = nixpkgs.lib;
    forAllSystems = genExpr: lib.genAttrs lib.systems.flakeExposed genExpr;
    nixpkgsFor = forAllSystems (system: import nixpkgs {inherit system;});
  in {
    formatter = forAllSystems (
      system:
        nixpkgsFor.${system}.alejandra
    );
    devShells = forAllSystems (
      system: let
        pkgs = nixpkgsFor.${system};
      in {
        default = pkgs.callPackage ({
          mkShell,
          python3,
        }:
          mkShell {
            packages = [
              (python3.withPackages (p:
                with p; [
                  numpy
                  matplotlib
                  sympy
                ]))
            ];
          }) {};
      }
    );
  };
}
