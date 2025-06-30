{ pkgs, lib, config, inputs, ... }:

{


  # https://devenv.sh/packages/
  packages = [
  pkgs.gnum4
  pkgs.gmp
  pkgs.mpfr
  pkgs.python3
  pkgs.maturin
  pkgs.gnumake
  pkgs.diffutils
  # pkgs.glibc
  ];

  languages.nix={enable=true;
    # components=["nixd"];
  };

  # https://devenv.sh/languages/
  languages.rust = {
      enable = true;
      # https://devenv.sh/reference/options/#languagesrustchannel
      channel = "stable";

      components = [ "rustc" "cargo" "clippy" "rustfmt" "rust-analyzer" ];
    };
  languages.python={
    enable=true;
    uv.enable = true;
  };
  languages.cplusplus.enable = true;


  # https://devenv.sh/git-hooks/
  # git-hooks.hooks.shellcheck.enable = true;

  # See full reference at https://devenv.sh/reference/options/
}
